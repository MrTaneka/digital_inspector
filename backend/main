"""
Digital Inspector Backend API
Детектирует подписи, печати и QR-коды на строительных документах
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import os
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime

# Создаем приложение
app = FastAPI(
    title="Digital Inspector API",
    description="API для автоматического обнаружения подписей, печатей и QR-кодов",
    version="1.0.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути к папкам
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"

# Создаем папки если их нет
for directory in [UPLOAD_DIR, OUTPUT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Подключаем статику для результатов
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Digital Inspector API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "process": "/api/process/{task_id}",
            "results": "/api/results/{task_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": check_gpu_availability()
    }


def check_gpu_availability() -> bool:
    """Проверка доступности GPU"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Загрузка документов для обработки
    Принимает: PDF и изображения (PNG, JPG, JPEG)
    """
    task_id = str(uuid.uuid4())
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        # Проверка типа файла
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            continue
        
        # Сохранение файла
        file_path = task_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_files.append({
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "type": file.content_type
        })
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    
    # Сохраняем метаданные
    metadata = {
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "files": uploaded_files,
        "status": "uploaded"
    }
    
    with open(task_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return {
        "task_id": task_id,
        "files_uploaded": len(uploaded_files),
        "files": uploaded_files
    }


@app.post("/api/process/{task_id}")
async def process_documents(task_id: str):
    """
    Обработка загруженных документов
    Запускает детекцию подписей, печатей и QR-кодов
    """
    task_dir = UPLOAD_DIR / task_id
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Загружаем метаданные
    with open(task_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Создаем папку для результатов
    results_dir = RESULTS_DIR / task_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Импортируем процессор
    from detector import DocumentDetector
    
    detector = DocumentDetector()
    
    all_results = []
    
    # Обрабатываем каждый файл
    for file_info in metadata["files"]:
        file_path = task_dir / file_info["filename"]
        
        try:
            result = detector.process_file(
                file_path=str(file_path),
                output_dir=str(results_dir)
            )
            all_results.append(result)
        except Exception as e:
            all_results.append({
                "filename": file_info["filename"],
                "error": str(e),
                "status": "error"
            })
    
    # Сохраняем результаты
    results_data = {
        "task_id": task_id,
        "processed_at": datetime.now().isoformat(),
        "results": all_results
    }
    
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    # Обновляем метаданные
    metadata["status"] = "processed"
    metadata["processed_at"] = datetime.now().isoformat()
    
    with open(task_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return {
        "task_id": task_id,
        "status": "completed",
        "results": all_results
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Получение результатов обработки"""
    results_dir = RESULTS_DIR / task_id
    results_file = results_dir / "results.json"
    
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


@app.get("/api/image/{task_id}/{filename}")
async def get_processed_image(task_id: str, filename: str):
    """Получение обработанного изображения"""
    image_path = RESULTS_DIR / task_id / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """Удаление задачи и всех связанных файлов"""
    task_dir = UPLOAD_DIR / task_id
    results_dir = RESULTS_DIR / task_id
    
    deleted = []
    
    if task_dir.exists():
        shutil.rmtree(task_dir)
        deleted.append("uploads")
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
        deleted.append("results")
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "deleted": deleted,
        "status": "success"
    }


@app.get("/api/tasks")
async def list_tasks():
    """Список всех задач"""
    tasks = []
    
    for task_dir in UPLOAD_DIR.iterdir():
        if task_dir.is_dir():
            metadata_file = task_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    tasks.append(metadata)
    
    return {
        "total": len(tasks),
        "tasks": sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
