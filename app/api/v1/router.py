from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool # <-- YENİ EKLENEN SİHİRLİ KOMUT
import json
from app.schemas.config import DataProcessingConfig, ModelTrainingConfig, ModelTestingConfig
from app.services.data_service import data_service
from app.services.ml_service import ml_service

api_router = APIRouter()

# --- 1. Veri İşleme ---
@api_router.post("/process-data")
async def process_data(file: UploadFile = File(...)):
    try:
        # Dosya okuma I/O işlemidir, async kalabilir
        content = await file.read()
        config_dict = json.loads(content)
        config = DataProcessingConfig(**config_dict)
        
        # CPU işlemi Threadpool'a gönderiliyor (Artık kilitlenmez)
        return await run_in_threadpool(data_service.process_data, config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Model Eğitimi (EN KRİTİK YER) ---
@api_router.post("/train-model")
async def train_model(file: UploadFile = File(...)):
    try:
        content = await file.read()
        config_dict = json.loads(content)
        config = ModelTrainingConfig(**config_dict)
        
        # Model eğitimi ağırdır, ana sistemi kilitlememesi için thread'e atıyoruz
        return await run_in_threadpool(ml_service.train_model, config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Model Testi ---
@api_router.post("/test-model")
async def test_model(file: UploadFile = File(...)):
    try:
        content = await file.read()
        config_dict = json.loads(content)
        config = ModelTestingConfig(**config_dict)
        
        return await run_in_threadpool(ml_service.test_model, config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))