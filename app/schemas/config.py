from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Dict, Any

# --- 1. Ortak Yapılar ---
class ModelConfig(BaseModel):
    type: Literal["random_forest", "xgboost", "neural_network"]
    params: Dict[str, Any]

# --- 2. Data Processing Config ---
class DataProcessingConfig(BaseModel):
    raw_data_path: str
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    random_state: int = 42
    output_train_path: str
    output_test_path: str

# --- 3. Training Config ---
class ModelTrainingConfig(BaseModel):
    train_data_path: str
    target_column: str
    feature_columns: List[str]
    
    # Veri alanının adı 'algorithm_config' olduğu için sorun yok
    algorithm_config: ModelConfig 
    
    experiment_name: str
    save_model_path: str
    
    # DÜZELTME: İsmi 'model_config' olmalı!
    model_config = ConfigDict(protected_namespaces=())

# --- 4. Testing Config ---
class ModelTestingConfig(BaseModel):
    test_data_path: str
    model_path: str
    model_type: Literal["random_forest", "xgboost", "neural_network"]
    target_column: str
    feature_columns: List[str]
    output_report_path: str
    
    # DÜZELTME: İsmi 'model_config' olmalı!
    model_config = ConfigDict(protected_namespaces=())