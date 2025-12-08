# app/api/v1/router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ml_service import ml_service
from app.schemas.config import MLPipelineConfig
import json

api_router = APIRouter()

@api_router.post("/train/config")
async def train_with_config(file: UploadFile = File(...)):
    """
    Config.json dosyasını yükle ve eğitimi başlat.
    """
    try:
        # 1. Dosyayı oku
        content = await file.read()
        config_dict = json.loads(content)
        
        # 2. Pydantic ile doğrula (Schema Validation)
        config = MLPipelineConfig(**config_dict)
        
        # 3. Servise yolla
        result = ml_service.train_from_config(config)
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Geçersiz JSON formatı.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))