from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from contextlib import asynccontextmanager
import os
import uuid
import json
import re
import time
import asyncio
import logging
import logging.config
from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from s_padf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "INFO",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":     "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level":     "DEBUG",
            "filename":  "app.log",
            "maxBytes":  10 * 1024 * 1024,
            "backupCount": 5,
            "encoding":  "utf-8",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class Job:
    job_id:        str
    file_path:     str
    analysis_type: int
    status:        JobStatus = JobStatus.QUEUED
    enqueued_at:   float     = field(default_factory=time.time)
    started_at:    float     = 0.0
    finished_at:   float     = 0.0
    result:        Any       = None
    error:         str       = ""


_job_store: dict[str, Job] = {}
_gpu_queue: asyncio.Queue  = asyncio.Queue()

MAX_PDF_PAGES   = None
JOB_TTL_SECONDS = 600
_BLANK_RESULT   = {"overview": "", "summary": "", "highlights": []}


# ---------------------------------------------------------------------------
# JSON helpers (unchanged)
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    match = re.search(rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_highlights_by_regex(text: str) -> list:
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    return [i.strip() for i in re.findall(r'"(.*?)"(?=\s*[,\]])', match.group(1), re.DOTALL) if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            for p in (item.split('", "') if '", "' in item else [item]):
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def _normalize_parsed(data, label: str = "") -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        if not data:
            return empty
        if isinstance(data[0], dict):
            merged: dict = {"overview": "", "summary": "", "highlights": []}
            for item in data:
                if not isinstance(item, dict):
                    continue
                if not merged["overview"] and item.get("overview"):
                    merged["overview"] = item["overview"]
                if not merged["summary"] and item.get("summary"):
                    merged["summary"] = item["summary"]
                merged["highlights"].extend(item.get("highlights") or [])
            logger.warning(f"extract_json {label}: got list of dicts — merged {len(data)} item(s)")
            return merged
        if isinstance(data[0], str):
            return {"overview": "", "summary": "", "highlights": [s for s in data if s]}
    logger.error(f"extract_json {label}: unexpected type {type(data).__name__}")
    return empty


def extract_json(text: str) -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text or not text.strip():
        return empty
    cleaned = _fix_json_string(text)
    try:
        return _postprocess_highlights(_normalize_parsed(json.loads(cleaned), "strategy-1"))
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            return _postprocess_highlights(_normalize_parsed(json.loads(candidate), "strategy-2"))
        except json.JSONDecodeError as e:
            repaired = re.sub(r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate)
            try:
                return _postprocess_highlights(_normalize_parsed(json.loads(repaired), "strategy-2-repair"))
            except json.JSONDecodeError:
                logger.warning(f"extract_json: strategy 2 repair failed — {e}")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}
    return empty


# ---------------------------------------------------------------------------
# Core pipeline (called by the GPU worker)
# ---------------------------------------------------------------------------

async def _process_job(job: Job) -> dict:
    rid       = job.job_id[:8]
    file_path = job.file_path
    t0        = time.perf_counter()

    try:
        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
        logger.info(f"[{rid}] pages={total_pages} analysing={pages_to_read}")

        try:
            pages = load_pdf(file_path, max_pages=pages_to_read)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if all_pages_blank(pages):
            logger.warning(f"[{rid}] blank PDF — skipping inference")
            result = dict(_BLANK_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            return result

        merged_text  = "\n\n".join(p.page_content for p in pages)
        final_output = await generate_analysis(merged_text)

        if was_truncated:
            final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

        logger.info(f"[{rid}] done in {time.perf_counter()-t0:.2f}s")
        return final_output

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{rid}] temp file deleted")


# ---------------------------------------------------------------------------
# Background GPU worker — drains queue one job at a time
# ---------------------------------------------------------------------------

async def _gpu_worker():
    logger.info("[gpu_worker] started")
    while True:
        job: Job       = await _gpu_queue.get()
        job.status     = JobStatus.PROCESSING
        job.started_at = time.time()
        wait           = job.started_at - job.enqueued_at
        logger.info(f"[gpu_worker] job {job.job_id[:8]} started (waited {wait:.1f}s)")

        try:
            job.result = await _process_job(job)
            job.status = JobStatus.DONE
        except HTTPException as e:
            job.error  = e.detail
            job.status = JobStatus.FAILED
            logger.error(f"[gpu_worker] job {job.job_id[:8]} HTTP error: {e.detail}")
        except Exception as e:
            job.error  = "Internal server error during PDF analysis."
            job.status = JobStatus.FAILED
            logger.exception(f"[gpu_worker] job {job.job_id[:8]} unhandled: {e}")
        finally:
            job.finished_at = time.time()
            _gpu_queue.task_done()
            logger.info(
                f"[gpu_worker] job {job.job_id[:8]} {job.status.value} "
                f"in {job.finished_at - job.started_at:.2f}s"
            )


# ---------------------------------------------------------------------------
# Cleanup worker — purges old completed jobs every 60s
# ---------------------------------------------------------------------------

async def _cleanup_worker():
    while True:
        await asyncio.sleep(60)
        now     = time.time()
        expired = [
            jid for jid, j in list(_job_store.items())
            if j.status in (JobStatus.DONE, JobStatus.FAILED)
            and (now - j.finished_at) > JOB_TTL_SECONDS
        ]
        for jid in expired:
            job = _job_store.pop(jid, None)
            if job and os.path.exists(job.file_path):
                os.remove(job.file_path)
        if expired:
            logger.info(f"[cleanup] purged {len(expired)} expired job(s)")


# ---------------------------------------------------------------------------
# Lifespan — start workers when server boots
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Upload folder ready: {os.path.abspath(UPLOAD_FOLDER)}")
    gpu_task     = asyncio.create_task(_gpu_worker())
    cleanup_task = asyncio.create_task(_cleanup_worker())
    yield
    gpu_task.cancel()
    cleanup_task.cancel()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /analyze — save file, enqueue, return job_id instantly (HTTP 202)
# ---------------------------------------------------------------------------

@app.post("/analyze", status_code=202)
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Submit a PDF for analysis.

    Returns HTTP 202 immediately with a job_id.
    Poll GET /status/{job_id} until status is 'done' or 'failed'.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.pdf")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"[{job_id[:8]}] saved {len(content):,} bytes")

    queue_depth = _gpu_queue.qsize()
    job = Job(job_id=job_id, file_path=file_path, analysis_type=analysis_type)
    _job_store[job_id] = job
    await _gpu_queue.put(job)
    logger.info(f"[{job_id[:8]}] enqueued at position {queue_depth}")

    return {
        "job_id":         job_id,
        "status":         JobStatus.QUEUED,
        "queue_position": queue_depth,
        "message": (
            "Processing will start shortly. Poll GET /status/{job_id} for results."
            if queue_depth == 0 else
            f"{queue_depth} job(s) ahead of yours. Poll GET /status/{{job_id}} for results."
        ),
    }


# ---------------------------------------------------------------------------
# GET /status/{job_id} — poll for result
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
async def get_status(
    job_id: str,
    analysis_type: int = Query(0, ge=0, le=3,
        description="Filter returned fields: 0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Poll job status.

    Responses by status:
      queued:     { status, queue_position, wait_seconds }
      processing: { status, elapsed_seconds }
      done:       { status, processing_time, result: {...} }
      failed:     { status, error }
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found. It may have expired (results kept for 10 minutes).")

    if job.status == JobStatus.QUEUED:
        ahead = sum(
            1 for j in _job_store.values()
            if j.status == JobStatus.QUEUED and j.enqueued_at < job.enqueued_at
        )
        return {"status": JobStatus.QUEUED, "queue_position": ahead,
                "wait_seconds": round(time.time() - job.enqueued_at, 1)}

    if job.status == JobStatus.PROCESSING:
        return {"status": JobStatus.PROCESSING,
                "elapsed_seconds": round(time.time() - job.started_at, 1)}

    if job.status == JobStatus.FAILED:
        return {"status": JobStatus.FAILED, "error": job.error}

    # DONE — apply analysis_type filter
    result = job.result or {}
    if analysis_type == 1:
        filtered = {"overview":   result.get("overview", "")}
    elif analysis_type == 2:
        filtered = {"summary":    result.get("summary", "")}
    elif analysis_type == 3:
        filtered = {"highlights": result.get("highlights", [])}
    else:
        filtered = result

    return {
        "status":          JobStatus.DONE,
        "processing_time": round(job.finished_at - job.started_at, 2),
        "result":          filtered,
    }


# ---------------------------------------------------------------------------
# GET /queue — admin overview of current queue state
# ---------------------------------------------------------------------------

@app.get("/queue")
async def queue_status():
    """Summary of all jobs currently tracked."""
    counts = {s.value: 0 for s in JobStatus}
    for j in _job_store.values():
        counts[j.status.value] += 1

    processing = next(
        ({"job_id": j.job_id, "elapsed_seconds": round(time.time() - j.started_at, 1)}
         for j in _job_store.values() if j.status == JobStatus.PROCESSING),
        None
    )

    return {
        "queue_depth":          _gpu_queue.qsize(),
        "jobs":                 counts,
        "currently_processing": processing,
    }