from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
from pdf2image import convert_from_bytes
import pytesseract
import re
from PIL import Image
from typing import List, Dict, Any
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os

# Router imports
from youtube_summary import router as yt_router
from materials import router as materials_router
from fastapi.staticfiles import StaticFiles

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract & Model
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model = SentenceTransformer("all-MiniLM-L6-v2")
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI app
app = FastAPI(
    title="Edu Materials AI Backend",
    description="FAQ extractor, video summarizer, and study material API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for study materials
app.mount("/study-materials", StaticFiles(directory="study_materials"), name="study-materials")

# Include routers
app.include_router(yt_router)
app.include_router(materials_router)

# Basic health
@app.get("/")
def root():
    return {"message": "📚 Edu Materials API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


# ---------------------------- PDF FAQ Extraction ----------------------------

@app.post("/analyze-papers/")
async def analyze_question_papers(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_questions_data = []
    processing_results = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            processing_results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": "Not a PDF file"
            })
            continue

        try:
            pdf_bytes = await file.read()
            questions = await process_pdf(pdf_bytes, file.filename)

            all_questions_data.extend(questions)
            processing_results.append({
                "filename": file.filename,
                "status": "success",
                "questions_found": len(questions)
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            processing_results.append({
                "filename": file.filename,
                "status": "error",
                "reason": str(e)
            })

    if not all_questions_data:
        return JSONResponse(
            status_code=200,
            content={
                "message": "No questions found in uploaded files",
                "processing_results": processing_results,
                "frequent_questions": []
            }
        )

    frequent_questions = await analyze_frequent_questions(all_questions_data)

    return {
        "message": f"Analyzed {len(files)} files",
        "total_questions_found": len(all_questions_data),
        "processing_results": processing_results,
        "frequent_questions": frequent_questions,
        "summary": {
            "total_files": len(files),
            "successful_files": len([r for r in processing_results if r["status"] == "success"]),
            "failed_files": len([r for r in processing_results if r["status"] == "error"]),
            "unique_question_groups": len(frequent_questions)
        }
    }

async def process_pdf(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    text = await asyncio.get_event_loop().run_in_executor(
        executor, extract_text_with_ocr, pdf_bytes
    )
    logger.info(f"📝 OCR for {filename} done")

    raw_questions = extract_questions(text)
    logger.info(f"🔍 {len(raw_questions)} questions found in {filename}")

    return [{
        "original": q,
        "normalized": normalize_question(q),
        "source_file": filename,
        "question_id": f"{filename}_{i+1}"
    } for i, q in enumerate(raw_questions)]

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(
        pdf_bytes,
        dpi=300,
        poppler_path=r"C:\Users\Pragati Kesharwani\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
    )
    text = ""
    for img in images:
        gray = img.convert("L")
        text += pytesseract.image_to_string(gray, config='--psm 6') + "\n"
    return text

def extract_questions(text: str) -> List[str]:
    lines = text.splitlines()
    questions = []

    question_starters = [
        'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whose',
        'define', 'explain', 'describe', 'state', 'write', 'discuss',
        'derive', 'prove', 'calculate', 'find', 'determine', 'show',
        'compare', 'differentiate', 'distinguish', 'analyze', 'evaluate',
        'solve', 'compute', 'obtain', 'draw', 'sketch', 'plot',
        'list', 'enumerate', 'mention', 'name', 'give', 'provide'
    ]

    imperative_patterns = [
        r'^(state|write|give|list|mention|name|draw|sketch)\b',
        r'^write\s+(a\s+)?(short\s+|brief\s+)?note\s+on\b',
        r'^give\s+(a\s+)?(brief\s+|short\s+)?account\s+of\b',
        r'^derive\s+the\s+(formula|equation|expression)\b',
        r'^prove\s+that\b',
        r'^show\s+that\b',
        r'^find\s+the\b',
        r'^calculate\s+the\b',
        r'^determine\s+the\b'
    ]

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5 or len(line) > 300:
            continue
        clean_line = re.sub(r'^[\d\)\.\-\*\+\s]+', '', line).strip()
        if not clean_line:
            continue

        if line.endswith('?'):
            questions.append(line)
            continue

        first_word = clean_line.split()[0].lower() if clean_line.split() else ""
        if first_word in question_starters:
            questions.append(line + ('?' if not line.endswith('?') else ''))
            continue

        for pattern in imperative_patterns:
            if re.search(pattern, clean_line.lower()):
                questions.append(line + ('?' if not line.endswith('?') else ''))
                break

    return questions

def normalize_question(question: str) -> str:
    q = question.lower().strip()
    if q.endswith('?'):
        q = q[:-1]
    q = re.sub(r'^[\d\)\.\-\*\+\s]+', '', q)
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'\b(19|20)\d{2}\b', '', q)
    q = re.sub(r'\b\d+\s*(marks?|points?)\b', '', q)
    q = re.sub(r'\b(briefly|short|detail|detailed|long)\b', '', q)
    q = re.sub(r'\b(note on|account of|discussion on)\b', '', q)
    q = re.sub(r'\b(with\s+)?(suitable\s+)?(neat\s+)?(diagram|figure|graph)\b', '', q)

    question_mappings = {
        r'^(what is|define|what do you mean by|what are)\b': 'what is',
        r'^(explain|describe|discuss|elaborate)\b': 'explain',
        r'^(state|mention|write|give)\b': 'state',
        r'^(how|in what way)\b': 'how',
        r'^(why|what is the reason)\b': 'why',
        r'^(derive|obtain|find the expression)\b': 'derive',
        r'^(prove|show that|demonstrate)\b': 'prove',
        r'^(calculate|compute|find|determine)\b': 'calculate'
    }

    for pattern, replacement in question_mappings.items():
        q = re.sub(pattern, replacement, q)

    q = re.sub(r'\s+', ' ', q).strip()
    if not q.endswith('?'):
        q += '?'

    return q

async def analyze_frequent_questions(questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not questions_data:
        return []

    normalized_questions = [q["normalized"] for q in questions_data]
    embeddings = model.encode(normalized_questions, convert_to_tensor=True)

    groups = []
    used_indices = set()
    similarity_threshold = 0.65

    for i, question_data in enumerate(questions_data):
        if i in used_indices:
            continue

        group = {
            "main_question": question_data["normalized"],
            "variants": [question_data["original"]],
            "source_files": [question_data["source_file"]],
            "frequency": 1,
            "question_ids": [question_data["question_id"]]
        }
        used_indices.add(i)

        for j in range(i + 1, len(questions_data)):
            if j in used_indices:
                continue
            similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
            if similarity >= similarity_threshold:
                group["variants"].append(questions_data[j]["original"])
                group["source_files"].append(questions_data[j]["source_file"])
                group["question_ids"].append(questions_data[j]["question_id"])
                group["frequency"] += 1
                used_indices.add(j)

        if group["frequency"] >= 2:
            group["source_files"] = list(set(group["source_files"]))
            group["unique_sources"] = len(group["source_files"])
            groups.append(group)

    groups.sort(key=lambda x: x["frequency"], reverse=True)
    for i, group in enumerate(groups):
        group["rank"] = i + 1

    logger.info(f"✅ Found {len(groups)} frequently asked question groups")
    return groups

from materials import router as materials_router
app.include_router(materials_router)
