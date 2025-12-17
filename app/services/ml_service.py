import pandas as pd
import os
import json
import mlflow # <-- MLOps Kütüphanesi
import mlflow.sklearn # <-- Sklearn modellerini kaydetmek için
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.schemas.config import ModelTrainingConfig, ModelTestingConfig
from app.services.model_factory import ModelFactory
from app.core.logging_config import logger

class MLService:
    
    # --- ENDPOINT 2: TRAIN (MLflow Entegre Edildi) ---
    def train_model(self, config: ModelTrainingConfig):
        logger.info(f"İşlem Başlıyor: Model Eğitimi - {config.experiment_name}")
        
        # 1. Dosya Kontrolü
        if not os.path.exists(config.train_data_path):
            logger.error(f"Dosya bulunamadı: {config.train_data_path}")
            raise FileNotFoundError(f"Train verisi bulunamadı: {config.train_data_path}")
            
        # 2. Veriyi Oku
        logger.info(f"Veri okunuyor: {config.train_data_path}")
        df = pd.read_csv(config.train_data_path)
        X = df[config.feature_columns]
        y = df[config.target_column]
        
        # 3. MLflow Deneyini Ayarla (Klasör gibi düşün)
        mlflow.set_experiment(config.experiment_name)

        # 4. MLOps Takibini Başlat
        with mlflow.start_run():
            # A) Parametreleri Logla (Hangi ayarlarla eğittik?)
            logger.info("Parametreler MLflow'a kaydediliyor...")
            mlflow.log_params(config.algorithm_config.params)
            mlflow.log_param("model_type", config.algorithm_config.type)
            mlflow.log_param("train_data_size", len(df))

            # B) Modeli Eğit
            model_instance = ModelFactory.get_model(config.algorithm_config)
            logger.info(f"Model ({config.algorithm_config.type}) eğitiliyor...")
            train_details = model_instance.train(X, y, config.algorithm_config.params)
            
            # C) Modeli Diske Kaydet (.pkl)
            os.makedirs(os.path.dirname(config.save_model_path), exist_ok=True)
            model_instance.save(config.save_model_path)
            logger.info(f"Model başarıyla diske kaydedildi: {config.save_model_path}")

            # D) Modeli MLflow'a Artifact Olarak Kaydet (Bulutta veya yerelde yedekleme)
            # Not: Sadece sklearn tabanlılar için örnekliyoruz, diğerleri için custom logic gerekebilir.
            if config.algorithm_config.type == "random_forest":
                mlflow.sklearn.log_model(model_instance.model, "model")
            
            logger.info("MLflow kaydı tamamlandı.")

            return {
                "status": "Training Completed",
                "experiment": config.experiment_name,
                "mlflow_run_id": mlflow.active_run().info.run_id, # <-- Run ID'yi dönüyoruz
                "save_path": config.save_model_path,
                "details": train_details
            }

    # --- ENDPOINT 3: TEST (Değişiklik Yok) ---
    def test_model(self, config: ModelTestingConfig):
        logger.info(f"Test işlemi başladı. Model: {config.model_path}")
        
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
        
        y_pred = model_instance.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        logger.info(f"Test metrikleri hesaplandı: Accuracy={metrics['accuracy']}")
        
        os.makedirs(os.path.dirname(config.output_report_path), exist_ok=True)
        with open(config.output_report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return {"status": "Test Completed", "metrics": metrics}