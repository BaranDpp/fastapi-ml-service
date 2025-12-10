import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.schemas.config import ModelTrainingConfig, ModelTestingConfig
from app.services.model_factory import ModelFactory

class MLService:
    
    # --- ENDPOINT 2: TRAIN ---
    def train_model(self, config: ModelTrainingConfig):
        if not os.path.exists(config.train_data_path):
            raise FileNotFoundError(f"Train verisi bulunamadı: {config.train_data_path}")
            
        print(f"Eğitim verisi okunuyor: {config.train_data_path}")
        df = pd.read_csv(config.train_data_path)
        
        X = df[config.feature_columns]
        y = df[config.target_column]
        
        # DEĞİŞİKLİK BURADA: config.algorithm_config kullanıyoruz
        model_instance = ModelFactory.get_model(config.algorithm_config)
        
        print(f"Model eğitiliyor ({config.algorithm_config.type})...")
        train_details = model_instance.train(X, y, config.algorithm_config.params)
        
        os.makedirs(os.path.dirname(config.save_model_path), exist_ok=True)
        model_instance.save(config.save_model_path)
        
        return {
            "status": "Training Completed",
            "model_type": config.algorithm_config.type,
            "save_path": config.save_model_path,
            "details": train_details
        }

    # --- ENDPOINT 3: TEST (Değişiklik yok, aynen kalıyor) ---
    def test_model(self, config: ModelTestingConfig):
        if not os.path.exists(config.test_data_path):
            raise FileNotFoundError(f"Test verisi yok: {config.test_data_path}")
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model yok: {config.model_path}")

        df = pd.read_csv(config.test_data_path)
        X_test = df[config.feature_columns]
        y_test = df[config.target_column]
        
        class TempConfig: type = config.model_type
        model_instance = ModelFactory.get_model(TempConfig)
        model_instance.load(config.model_path)
        
        print(f"Test yapılıyor... Model: {config.model_type}")
        y_pred = model_instance.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        os.makedirs(os.path.dirname(config.output_report_path), exist_ok=True)
        with open(config.output_report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return {"status": "Test Completed", "metrics": metrics}

ml_service = MLService()