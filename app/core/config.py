from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Baran_Polat ML_FastAPI"
    VERSION: str = "1.0.0"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH: str = os.path.join(BASE_DIR, "data", "ai-task-2-zscore-processed.csv")
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "benim_modelim.keras")
    SCALER_PATH: str = os.path.join(BASE_DIR, "models", "benim_scaler.pkl")
    class Config:
        case_sensitive = True
settings = Settings()