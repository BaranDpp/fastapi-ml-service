from pydantic import BaseModel
from typing import List, Optional

class PredictionInput(BaseModel):
    features: List[float]  # Örneğin: [0.5, 1.2, -0.3, ...]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.1, 0.5, 1.2, 0.8, -0.5] # Örnek değerler
            }
        }

class PredictionOutput(BaseModel):
    prediction: float
    class_label: int  # 0 veya 1
    confidence: float # Güven oranı

class TrainOutput(BaseModel):
    message: str
    accuracy: float
    loss: float