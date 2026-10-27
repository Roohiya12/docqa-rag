"""Load raw documents from a folder into (doc_id, text) pairs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def load_pdf_file(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(raw_dir: Path) -> List[Tuple[str, str]]:
    """Return [(doc_id, text)] for every .txt/.md/.pdf file in raw_dir."""
    docs: List[Tuple[str, str]] = []
    for path in sorted(Path(raw_dir).glob("*")):
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = load_pdf_file(path)
        else:
            continue
        if text.strip():
            docs.append((path.stem, text))
    return docs
