import pandas as pd
import os
from sklearn.model_selection import train_test_split
from app.schemas.config import DataProcessingConfig
from app.core.logging_config import logger  # <-- Logger eklendi

class DataService:
    def process_data(self, config: DataProcessingConfig):
        logger.info(f"Veri işleme süreci başladı: {config.raw_data_path}")
        
        # 1. Dosya Kontrolü
        if not os.path.exists(config.raw_data_path):
            error_msg = f"Ham veri dosyası bulunamadı: {config.raw_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if config.raw_data_path.endswith('.csv'):
            df = pd.read_csv(config.raw_data_path)
        elif config.raw_data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(config.raw_data_path)
        else:
            raise ValueError("Geçersiz dosya formatı. Lütfen .csv veya .xlsx dosyası kullanın.")
        
        # 2. Güvenlik Kontrolleri
        missing_features = [col for col in config.feature_columns if col not in df.columns]
        if missing_features:
            error_msg = f"Şu sütunlar CSV'de yok: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if config.target_column not in df.columns:
            error_msg = f"Hedef sütun '{config.target_column}' CSV dosyasında yok!"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Target yanlışlıkla feature listesindeyse çıkar
        if config.target_column in config.feature_columns:
            logger.warning(f"Target '{config.target_column}' feature listesinden çıkarılıyor.")
            config.feature_columns = [col for col in config.feature_columns if col != config.target_column]

        # 3. Temizlik
        df = df.dropna(subset=[config.target_column])
        if len(df) == 0:
            error_msg = "Temizlik sonrası veri kalmadı! Hedef sütun tamamen boş."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 4. Seçim ve Bölme
        X = df[config.feature_columns]
        y = df[config.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.test_size, 
            random_state=config.random_state
        )
        
        # 5. Kaydetme
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        os.makedirs(os.path.dirname(config.output_train_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.output_test_path), exist_ok=True)
        
        train_df.to_csv(config.output_train_path, index=False)
        test_df.to_csv(config.output_test_path, index=False)
        
        logger.info(f"Veri işleme tamamlandı. Train: {len(train_df)} satır, Test: {len(test_df)} satır.")
        
        return {
            "status": "Data Processed",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_path": config.output_train_path
        }