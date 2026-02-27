from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import docx  # python-docx
import pdfplumber
from PIL import Image
import pytesseract


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def extract_text_from_txt(file_obj: BinaryIO) -> str:
    return file_obj.read().decode("utf-8", errors="ignore")


def extract_text_from_docx(file_obj: BinaryIO) -> str:
    tmp_path = Path("tmp_upload.docx")
    tmp_path.write_bytes(file_obj.read())
    doc = docx.Document(str(tmp_path))
    text = "\n".join(p.text for p in doc.paragraphs)
    tmp_path.unlink(missing_ok=True)
    return text.strip()


def extract_text_from_image(file_obj: BinaryIO) -> str:
    img = Image.open(file_obj)
    txt = pytesseract.image_to_string(img)
    return txt.strip()

