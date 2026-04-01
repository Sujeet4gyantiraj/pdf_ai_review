import os
from pathlib import Path

OUTPUT_FILE = "project_structure.md"

# Directories to ignore anywhere in project
EXCLUDED_DIRS = {
    ".git",
    ".vscode",
    "__pycache__",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
}

# Files to ignore
EXCLUDED_FILES = {
    OUTPUT_FILE,
    "db.sqlite3",
    ".env",
    "README.md",
    ".htaccess",
    "app.log",
    "uvicorn.log",
    "gunicorn.log"
}

# Binary file extensions
BINARY_EXTENSIONS = {
    ".png",".jpg",".jpeg",".gif",".ico",".svg",".webp",
    ".woff",".woff2",".ttf",".eot",".otf",
    ".pdf",".zip",".gz",".tar",".rar",
    ".exe",".dll",".so",".a",".lib",".jar",".mp3"
}


def is_binary_file(file_path: Path):
    return file_path.suffix.lower() in BINARY_EXTENSIONS


def build_tree(start_path: Path):
    tree_lines = []

    for root, dirs, files in os.walk(start_path):

        # Remove excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        level = root.replace(str(start_path), "").count(os.sep)
        indent = "│   " * level

        folder_name = os.path.basename(root) if level != 0 else "."
        tree_lines.append(f"{indent}├── {folder_name}/")

        sub_indent = "│   " * (level + 1)

        for f in files:
            if f in EXCLUDED_FILES:
                continue
            tree_lines.append(f"{sub_indent}├── {f}")

    return "\n".join(tree_lines)


def collect_files(start_path: Path):
    collected = []

    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for f in files:
            if f in EXCLUDED_FILES:
                continue

            file_path = Path(root) / f

            if is_binary_file(file_path):
                continue

            collected.append(file_path)

    return collected


def main():
    project_root = Path(".")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        out.write("# Django Project Structure\n\n")

        out.write("```\n")
        out.write(build_tree(project_root))
        out.write("\n```\n\n")

        out.write("# File Contents\n\n")

        for file_path in collect_files(project_root):

            rel_path = file_path.relative_to(project_root)
            extension = file_path.suffix.replace(".", "") or "text"

            out.write("---\n")
            out.write(f"File: {rel_path}\n")
            out.write("---\n\n")

            out.write(f"```{extension}\n")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    out.write(f.read())
            except:
                out.write("<< Could not read file >>")

            out.write("\n```\n\n")

    print(f"✅ Django project exported to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
