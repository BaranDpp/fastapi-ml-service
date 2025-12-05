# app/api/v1/router.py
from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionInput, PredictionOutput, TrainOutput # <-- app ile başlamalı
from app.services.ml_service import ml_service # <-- app ile başlamalı

api_router = APIRouter()

@api_router.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Modeli kullanarak tahmin yapar.
    """
    try:
        result = ml_service.predict(input_data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train", response_model=TrainOutput)
def train_model():
    """
    Model eğitimini tetikler.
    Not: Gerçek hayatta bu işlem uzun süreceği için BackgroundTasks ile asenkron yapılmalıdır.
    """
    try:
        metrics = ml_service.train()
        return {
            "message": "Model başarıyla eğitildi ve yüklendi.",
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Veri seti (csv) bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eğitim hatası: {str(e)}")