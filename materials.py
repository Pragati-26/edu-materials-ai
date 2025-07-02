from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
from typing import List

router = APIRouter()
BASE_DIR = os.path.join(os.path.dirname(__file__), "study_materials")


@router.get("/materials/")
def list_classes():
    try:
        return {"classes": [cls for cls in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, cls))]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/materials/{class_id}")
def list_subjects(class_id: str):
    class_path = os.path.join(BASE_DIR, class_id)
    if not os.path.exists(class_path):
        raise HTTPException(status_code=404, detail="Class not found")
    return {"subjects": [s for s in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, s))]}


@router.get("/materials/{class_id}/{subject}")
def list_materials(class_id: str, subject: str):
    subject_path = os.path.join(BASE_DIR, class_id, subject)
    if not os.path.exists(subject_path):
        raise HTTPException(status_code=404, detail="Subject not found")

    files = os.listdir(subject_path)
    return {
        "materials": [
            {
                "name": file,
                "url": f"/study-materials/{class_id}/{subject}/{file}"
            }
            for file in files if os.path.isfile(os.path.join(subject_path, file))
        ]
    }


@router.post("/materials/{class_id}/{subject}/upload")
def upload_file(class_id: str, subject: str, file: UploadFile = File(...)):
    upload_path = os.path.join(BASE_DIR, class_id, subject)
    os.makedirs(upload_path, exist_ok=True)
    filepath = os.path.join(upload_path, file.filename)
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return {"message": "Upload successful", "filename": file.filename}
