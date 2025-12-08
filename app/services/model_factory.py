# app/services/model_factory.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from app.schemas.config import ModelConfig

# 1. Soyut Temel Sınıf (Interface)
# Tüm modeller bu kurallara uymak ZORUNDA.
class BaseMLModel(ABC):
    @abstractmethod
    def train(self, X, y, params: dict):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass

# 2. Concrete Class: Random Forest
class RandomForestAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        # JSON'dan gelen parametreleri unpack ediyoruz
        self.model = RandomForestClassifier(**params)
        self.model.fit(X, y)
        return {"status": "trained", "algorithm": "RandomForest"}

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)

# 3. Concrete Class: XGBoost
class XGBoostAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        return {"status": "trained", "algorithm": "XGBoost"}

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path.replace('.pkl', '.json'))

# 4. Concrete Class: Neural Network
class NeuralNetworkAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        input_dim = X.shape[1]
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(input_dim,)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid') # Binary classification varsayımı
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # NN parametrelerini ayıkla (epochs, batch_size)
        epochs = params.get('epochs', 10)
        batch_size = params.get('batch_size', 32)
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return {"status": "trained", "algorithm": "NeuralNetwork"}

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def save(self, path):
        # Keras .keras uzantısı ister
        save_path = path.replace('.pkl', '.keras')
        self.model.save(save_path)

# 5. THE FACTORY (Fabrika)
class ModelFactory:
    @staticmethod
    def get_model(config: ModelConfig) -> BaseMLModel:
        if config.type == "random_forest":
            return RandomForestAdapter()
        elif config.type == "xgboost":
            return XGBoostAdapter()
        elif config.type == "neural_network":
            return NeuralNetworkAdapter()
        else:
            raise ValueError(f"Bilinmeyen model tipi: {config.type}")