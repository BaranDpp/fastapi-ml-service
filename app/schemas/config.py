from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Dict, Any, Optional

# --- 1. Ortak Yapılar ---
class ModelConfig(BaseModel):
    type: Literal["random_forest", "xgboost", "neural_network"]
    params: Dict[str, Any]

# --- 2. Data Processing Config ---
class DataProcessingConfig(BaseModel):
    # BURASI DEĞİŞTİ: Alanı geri getirdik ama Optional yaptık (Boş geçilebilir)
    raw_data_path: Optional[str] = None 
    
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    random_state: int = 42
    output_train_path: str = "data/processed_train.csv"
    output_test_path: str = "data/processed_test.csv"

# --- 3. Training Config ---
class ModelTrainingConfig(BaseModel):
    # BURASI DEĞİŞTİ: Alanı geri getirdik ama Optional yaptık
    train_data_path: Optional[str] = None
    
    target_column: str
    feature_columns: List[str]
    algorithm_config: ModelConfig
    experiment_name: str
    save_model_path: str = "models/model.pkl"
    
    model_config = ConfigDict(protected_namespaces=())

# --- 4. Testing Config ---
class ModelTestingConfig(BaseModel):
    # BURASI DEĞİŞTİ: Alanları geri getirdik ama Optional yaptık
    test_data_path: Optional[str] = None
    model_path: Optional[str] = None
    
    model_type: Literal["random_forest", "xgboost", "neural_network"]
    target_column: str
    feature_columns: List[str]
    output_report_path: str = "reports/test_results.json"
    
    model_config = ConfigDict(protected_namespaces=())