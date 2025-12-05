# app/services/ml_service.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from app.core.config import settings

class MLService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        print("Artifacts yükleniyor...")
        if os.path.exists(settings.MODEL_PATH):
            self.model = load_model(settings.MODEL_PATH)
            print(f"Model yüklendi: {settings.MODEL_PATH}")
        
        if os.path.exists(settings.SCALER_PATH):
            self.scaler = joblib.load(settings.SCALER_PATH)
            print(f"Scaler yüklendi: {settings.SCALER_PATH}")

    def train(self):
        print(f"Veri okunuyor: {settings.DATA_PATH}...")
        df = pd.read_csv(settings.DATA_PATH)
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        os.makedirs(os.path.dirname(settings.SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, settings.SCALER_PATH)
        
        input_dim = X_train.shape[1]
        
        model = Sequential([
            Dense(16, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            ModelCheckpoint(settings.MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = load_model(settings.MODEL_PATH)
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        return {"loss": loss, "accuracy": accuracy}

    def predict(self, features: list):
        if not self.model or not self.scaler:
            raise Exception("Model veya Scaler yüklü değil! Önce /train endpointini çalıştırın.")
        
        input_data = np.array(features).reshape(1, -1)
        scaled_data = self.scaler.transform(input_data)
        
        prediction_prob = self.model.predict(scaled_data)[0][0]
        prediction_class = int(prediction_prob > 0.5)
        
        return {
            "prediction": float(prediction_prob),
            "class_label": prediction_class,
            "confidence": float(prediction_prob if prediction_class == 1 else 1 - prediction_prob)
        }

ml_service = MLService()