import os
import wave
import uuid
import logging
from pathlib import Path
from datetime import datetime

import pyaudio

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("audio_recorder")

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR        = Path("audiofiles")          # folder created next to this script
CHANNELS        = 1                            # mono
SAMPLE_RATE     = 16000                        # 16 kHz — ideal for Whisper
CHUNK           = 1024                         # frames per buffer
FORMAT          = pyaudio.paInt16              # 16-bit PCM
RECORD_SECONDS  = int(os.getenv("RECORD_SECONDS", "5"))   # default 5 s


def ensure_save_dir() -> Path:
    """Create the audiofiles/ directory if it doesn't exist."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Save directory: %s", SAVE_DIR.resolve())
    return SAVE_DIR


def build_filename() -> str:
    """
    Unique filename:  YYYYMMDD_HHMMSS_<uuid8>.wav
    e.g.  20240315_143022_c3f1a2b3.wav
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id  = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_id}.wav"


def list_input_devices(pa: pyaudio.PyAudio) -> None:
    """Print all available input devices for debugging."""
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")
    print()


def record(duration: int = RECORD_SECONDS, device_index: int | None = None) -> Path:
    """
    Record audio from the microphone for `duration` seconds,
    save as a WAV file inside audiofiles/, and return the saved path.

    Parameters
    ----------
    duration     : recording length in seconds
    device_index : PyAudio device index to use (None = system default)
    """
    save_dir  = ensure_save_dir()
    filename  = build_filename()
    save_path = save_dir / filename

    pa = pyaudio.PyAudio()

    try:
        list_input_devices(pa)

        logger.info("Opening microphone (device=%s, rate=%d Hz, channels=%d) ...",
                    device_index or "default", SAMPLE_RATE, CHANNELS)

        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

        print(f"🎙  Recording for {duration} second(s) ...  Press Ctrl+C to stop early.")
        frames: list[bytes] = []

        total_chunks = int(SAMPLE_RATE / CHUNK * duration)
        for i in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Simple progress bar
            pct  = int((i + 1) / total_chunks * 40)
            bar  = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {int((i + 1) / total_chunks * 100):3d}%", end="", flush=True)

        print("\n✅  Recording complete.")

    except KeyboardInterrupt:
        print("\n⚠️   Recording stopped early by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # ── Save as WAV ───────────────────────────────────────────────────────────
    with wave.open(str(save_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    size_kb = save_path.stat().st_size / 1024
    logger.info("Saved → %s  (%.1f KB)", save_path.resolve(), size_kb)
    print(f"\n📁  File saved: {save_path.resolve()}  ({size_kb:.1f} KB)")

    return save_path


# A farmer and his wife had a baby and a pet mongoose. One day, leaving the baby with the mongoose, the wife went to the market. Upon returning, she saw the mongoose’s mouth covered in blood. Thinking it had killed her son, she killed it instantly. Inside, she found the baby safe beside a dead snake, realizing the mongoose had saved


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record mic audio and save to audiofiles/")
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=RECORD_SECONDS,
        help=f"Recording duration in seconds (default: {RECORD_SECONDS})",
    )
    parser.add_argument(
        "--device", "-D",
        type=int,
        default=None,
        help="Input device index (default: system default). Run without --device to list devices.",
    )
    args = parser.parse_args()

    saved = record(duration=args.duration, device_index=args.device)
    print(f"\n🎧  Done! Recorded file: {saved}")