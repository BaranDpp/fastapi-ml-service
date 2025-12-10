from abc import ABC, abstractmethod
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf 
from app.schemas.config import ModelConfig

# --- Abstract Base Class ---
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
    
    @abstractmethod
    def load(self, path: str):
        pass

# --- 1. Random Forest Adapter ---
class RandomForestAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        self.model = RandomForestClassifier(**params)
        self.model.fit(X, y)
        return {"algorithm": "RandomForest", "params": params}

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)

# --- 2. XGBoost Adapter ---
class XGBoostAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        return {"algorithm": "XGBoost", "params": params}

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        if not path.endswith(".json"):
            path = path.replace(".pkl", ".json")
        self.model.save_model(path)
        
    def load(self, path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

# --- 3. Neural Network Adapter ---
class NeuralNetworkAdapter(BaseMLModel):
    def __init__(self):
        self.model = None

    def train(self, X, y, params: dict):
        input_dim = X.shape[1]
        
        # TensorFlow Modern Çağırım Şekli
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        epochs = params.get('epochs', 10)
        batch_size = params.get('batch_size', 32)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return {"algorithm": "NeuralNetwork", "epochs": epochs}

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def save(self, path):
        if not path.endswith(".keras"):
            path = path.replace(".pkl", ".keras")
        self.model.save(path)
        
    def load(self, path):
        self.model = tf.keras.models.load_model(path)

# --- THE FACTORY ---
class ModelFactory:
    @staticmethod
    def get_model(config) -> BaseMLModel:
        # Config objesi mi yoksa dict mi geldi kontrolü
        model_type = config.type if hasattr(config, 'type') else config['type']
        
        if model_type == "random_forest":
            return RandomForestAdapter()
        elif model_type == "xgboost":
            return XGBoostAdapter()
        elif model_type == "neural_network":
            return NeuralNetworkAdapter()
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")