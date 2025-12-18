from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
import json
import os
import shutil
import pandas as pd
from typing import Optional
from app.schemas.config import DataProcessingConfig, ModelTrainingConfig, ModelTestingConfig
from app.services.data_service import DataService
from app.services.ml_service import MLService
from app.core.logging_config import logger

api_router = APIRouter()

# Dependency Injection
def get_data_service() -> DataService:
    return DataService()

def get_ml_service() -> MLService:
    return MLService()

# --- YARDIMCI FONKSİYONLAR ---
def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

def generate_auto_config(file_path: str, target_col: str, config_type: str):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=1)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path, nrows=1)
    else:
        raise ValueError("Desteklenmeyen format! Sadece .csv veya .xlsx")
    
    if target_col not in df.columns:
        raise ValueError(f"Hedef sütun '{target_col}' dosyada bulunamadı! Mevcut sütunlar: {list(df.columns)}")
    
    features = [col for col in df.columns if col != target_col]
    
    if config_type == "process":
        return {"target_column": target_col, "feature_columns": features, "test_size": 0.2, "raw_data_path": file_path}
    elif config_type == "train":
        return {"experiment_name": "Auto_Experiment", "target_column": target_col, "feature_columns": features, "algorithm_config": {"type": "random_forest", "params": {"n_estimators": 100}}, "train_data_path": file_path, "save_model_path": "models/auto_model.pkl"}

# --- 0. Health Check (YENİ - Yöneticiler buna bayılır) ---
@api_router.get(
    "/health", 
    tags=["System Status"],
    summary="🚑 Sistem Sağlık Kontrolü",
    description="API'nin ayakta olup olmadığını ve versiyon bilgisini döner."
)
async def health_check():
    return {"status": "active", "version": "2.0.0", "mode": "production"}

# --- 1. Veri İşleme ---
@api_router.post(
    "/process-data", 
    response_class=FileResponse,
    tags=["1. Data Pipeline"],
    summary="🧹 Veri Temizleme ve Bölme",
    description="CSV veya Excel dosyası yükleyin. Sistem otomatik olarak temizleyip, Train/Test olarak ayırıp size geri indirtecektir."
)
async def process_data(
    file: UploadFile = File(..., description="İşlenecek ham veri dosyası (.csv veya .xlsx)"),
    config_str: Optional[str] = Form(None, description="Opsiyonel: Detaylı JSON ayarları"),
    target_column: Optional[str] = Form(None, description="JSON yoksa, sadece hedef sütun adını yazın (örn: 'price')"),
    service: DataService = Depends(get_data_service)
):
    try:
        temp_input_path = f"data/uploads/{file.filename}"
        save_upload_file(file, temp_input_path)
        
        if config_str:
            logger.info("Manuel config kullanılıyor.")
            config_dict = json.loads(config_str)
        else:
            if not target_column:
                raise HTTPException(status_code=400, detail="Ya JSON config gönderin ya da 'target_column' alanını doldurun.")
            logger.info("Otomatik config üretiliyor...")
            config_dict = generate_auto_config(temp_input_path, target_column, "process")

        config_obj = DataProcessingConfig(**config_dict)
        if not config_obj.raw_data_path:
            config_obj.raw_data_path = temp_input_path
            
        result = await run_in_threadpool(service.process_data, config_obj)
        
        return FileResponse(path=result["train_path"], filename="processed_train_data.csv", media_type='text/csv')
        
    except Exception as e:
        logger.error(f"Hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Model Eğitimi ---
@api_router.post(
    "/train-model", 
    response_class=FileResponse,
    tags=["2. Machine Learning"],
    summary="🧠 Model Eğitimi (AutoML)",
    description="İşlenmiş veriyi yükleyin. Random Forest, XGBoost vb. ile eğitilen model (.pkl) otomatik olarak iner. MLflow ile loglanır."
)
async def train_model(
    file: UploadFile = File(..., description="Eğitim verisi (CSV/Excel)"), 
    config_str: Optional[str] = Form(None, description="Model hiperparametreleri (JSON)"),
    target_column: Optional[str] = Form(None, description="JSON yoksa hedef sütun adı"),
    service: MLService = Depends(get_ml_service)
):
    try:
        temp_train_path = f"data/uploads/train_{file.filename}"
        save_upload_file(file, temp_train_path)
        
        if config_str:
            config_dict = json.loads(config_str)
        else:
            if not target_column:
                raise HTTPException(status_code=400, detail="Target column veya JSON gerekli.")
            config_dict = generate_auto_config(temp_train_path, target_column, "train")
        
        config_obj = ModelTrainingConfig(**config_dict)
        if not config_obj.train_data_path:
            config_obj.train_data_path = temp_train_path
        
        result = await run_in_threadpool(service.train_model, config_obj)
        
        return FileResponse(path=result["save_path"], filename="trained_model.pkl", media_type='application/octet-stream')
        
    except Exception as e:
        logger.error(f"Train Hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Model Testi ---
@api_router.post(
    "/test-model",
    tags=["2. Machine Learning"],
    summary="📊 Model Performans Testi",
    description="Test verisini ve eğitilmiş model (.pkl) dosyasını yükleyin. Size Accuracy, F1 Score gibi metrikleri içeren bir rapor döner."
)
async def test_model(
    test_file: UploadFile = File(..., description="Test verisi"),
    model_file: UploadFile = File(..., description="Eğitilmiş model dosyası"),
    config_str: str = Form(..., description="Config JSON"),
    service: MLService = Depends(get_ml_service)
):
    try:
        config_dict = json.loads(config_str)
        temp_test_path = f"data/uploads/test_{test_file.filename}"
        temp_model_path = f"models/uploads/{model_file.filename}"
        
        save_upload_file(test_file, temp_test_path)
        save_upload_file(model_file, temp_model_path)
        
        config_obj = ModelTestingConfig(**config_dict)
        setattr(config_obj, 'test_data_path', temp_test_path)
        setattr(config_obj, 'model_path', temp_model_path)
        
        logger.info("Test Model isteği alındı.")
        return await run_in_threadpool(service.test_model, config_obj)
        
    except Exception as e:
        logger.error(f"Test Hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))