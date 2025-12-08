# app/services/ml_service.py
import pandas as pd
from app.schemas.config import MLPipelineConfig
from app.services.model_factory import ModelFactory
import os

class MLService:
    def train_from_config(self, config: MLPipelineConfig):
        # 1. Veriyi Oku (Config'deki path'ten)
        # Not: Gerçek hayatta bu dosyalar S3'ten veya veritabanından gelebilir.
        if not os.path.exists(config.data.train_csv_path):
            raise FileNotFoundError(f"CSV bulunamadı: {config.data.train_csv_path}")
            
        df = pd.read_csv(config.data.train_csv_path)
        
        # 2. Preprocessing (Basitçe feature seçimi)
        X = df[config.data.feature_columns]
        y = df[config.data.target_column]
        
        # 3. Factory'den doğru modeli al
        model_instance = ModelFactory.get_model(config.model)
        
        # 4. Eğit (Polimorfizm sayesinde hepsi .train() metoduna sahip)
        result = model_instance.train(X, y, config.model.params)
        
        # 5. Kaydet
        os.makedirs(os.path.dirname(config.training.save_model_path), exist_ok=True)
        model_instance.save(config.training.save_model_path)
        
        return {
            "experiment": config.experiment_name,
            "model_type": config.model.type,
            "result": result
        }

ml_service = MLService()