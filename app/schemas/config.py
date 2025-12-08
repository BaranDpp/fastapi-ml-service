# app/schemas/config.py
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

class DataConfig(BaseModel):
    train_csv_path: str
    test_csv_path: str
    target_column: str
    feature_columns: List[str]
    shuffle: bool = True
    test_size: float = 0.2

class ModelConfig(BaseModel):
    # Literal sayesinde sadece bu 3 değer gelebilir, yoksa hata verir.
    type: Literal["random_forest", "xgboost", "neural_network"]
    params: Dict[str, Any] # Esnek parametreler (epoch, depth vs.)

class TrainingConfig(BaseModel):
    cv_folds: int = 5
    save_model_path: str

class MLPipelineConfig(BaseModel):
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig