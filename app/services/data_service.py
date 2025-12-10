import pandas as pd
import os
from sklearn.model_selection import train_test_split
from app.schemas.config import DataProcessingConfig

class DataService:
    def process_data(self, config: DataProcessingConfig):
        # 1. Dosya Kontrolü
        if not os.path.exists(config.raw_data_path):
            raise FileNotFoundError(f"Ham veri dosyası bulunamadı: {config.raw_data_path}")
        
        print(f"Veri işleniyor: {config.raw_data_path}")
        df = pd.read_csv(config.raw_data_path)
        
        # 2. Güvenlik Kontrolleri
        missing_features = [col for col in config.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"HATA: Şu sütunlar CSV'de yok: {missing_features}")

        if config.target_column not in df.columns:
            raise ValueError(f"HATA: Hedef sütun '{config.target_column}' CSV dosyasında yok!")

        # Target yanlışlıkla feature listesindeyse çıkar
        if config.target_column in config.feature_columns:
            config.feature_columns = [col for col in config.feature_columns if col != config.target_column]

        # 3. Temizlik
        df = df.dropna(subset=[config.target_column])
        if len(df) == 0:
            raise ValueError("HATA: Temizlik sonrası veri kalmadı!")

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
        
        return {
            "status": "Data Processed",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_path": config.output_train_path
        }

data_service = DataService()