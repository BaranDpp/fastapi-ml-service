from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.concurrency import run_in_threadpool
import json
from app.schemas.config import DataProcessingConfig, ModelTrainingConfig, ModelTestingConfig
from app.services.data_service import DataService
from app.services.ml_service import MLService
from app.core.logging_config import logger

api_router = APIRouter()

# --- DEPENDENCY INJECTION (BAĞIMLILIK ENJEKSİYONU) ---
# Bu fonksiyonlar her istekte servisin yeni veya temiz bir kopyasını verir.
def get_data_service() -> DataService:
    return DataService()

def get_ml_service() -> MLService:
    return MLService()

# --- 1. Veri İşleme ---
@api_router.post("/process-data")
async def process_data(
    file: UploadFile = File(...),
    service: DataService = Depends(get_data_service) # <-- Senior Dokunuş
):
    try:
        content = await file.read()
        config_dict = json.loads(content)
        config = DataProcessingConfig(**config_dict)
        
        logger.info("Process Data isteği alındı.")
        return await run_in_threadpool(service.process_data, config)
        
    except Exception as e:
        logger.error(f"Process Data Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Model Eğitimi ---
@api_router.post("/train-model")
async def train_model(
    file: UploadFile = File(...),
    service: MLService = Depends(get_ml_service) # <-- Senior Dokunuş
):
    try:
        content = await file.read()
        config_dict = json.loads(content)
        config = ModelTrainingConfig(**config_dict)
        
        logger.info(f"Train Model isteği: {config.experiment_name}")
        return await run_in_threadpool(service.train_model, config)
        
    except Exception as e:
        logger.error(f"Train Model Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Model Testi ---
@api_router.post("/test-model")
async def test_model(
    file: UploadFile = File(...),
    service: MLService = Depends(get_ml_service) # <-- Senior Dokunuş
):
    try:
        content = await file.read()
        config_dict = json.loads(content)
        config = ModelTestingConfig(**config_dict)
        
        logger.info("Test Model isteği alındı.")
        return await run_in_threadpool(service.test_model, config)
        
    except Exception as e:
        logger.error(f"Test Model Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))