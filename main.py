"""FastAPI backend exposing the plagiarism scan pipeline."""
import shutil
import tempfile
import os
import sys

# allow running this file directly with `uvicorn backend.api.main:app`
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.pipeline import scan_document

app = FastAPI(title="Explainable Plagiarism Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying for real
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/scan")
async def scan(file: UploadFile):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        report = scan_document(tmp_path)
    finally:
        os.remove(tmp_path)

    return report


# Run with: uvicorn backend.api.main:app --reload --port 8000
