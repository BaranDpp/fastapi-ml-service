# 🚀 FastAPI Config-Driven ML Pipeline

This project is a production-ready, **End-to-End Machine Learning Backend** built with **FastAPI**. It allows users to manage the entire ML lifecycle (Data Processing, Training, Testing) using dynamic **JSON configuration files**, without changing the source code.

The system is designed with **Clean Architecture** principles, **SOLID** patterns, and includes a robust **Factory Pattern** to support multiple algorithms dynamically.

## 🌟 Key Features

- **End-to-End Pipeline:** Dedicated endpoints for Data Processing, Model Training, and Testing.
- **🏭 Factory Design Pattern:** Dynamically supports **Random Forest**, **XGBoost**, and **Neural Networks** (TensorFlow/Keras).
- **📄 Config-Driven:** Control hyperparameters, features, and file paths via JSON uploads.
- **⚡ Async & Non-Blocking:** Heavy ML operations run in a threadpool, ensuring the API remains responsive.
- **🛡️ Type Safety:** Strict validation using **Pydantic V2**.
- **Container Ready:** Clean dependency management with `requirements.txt`.

## 🛠️ Tech Stack

- **Framework:** FastAPI, Uvicorn
- **ML Engine:** TensorFlow (Keras), XGBoost, Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Architecture:** Layered Architecture (Router, Service, Schema, Factory)

## 📂 Project Structure
fastapi_ml_project/
├── app/
│   ├── api/v1/router.py          # API Endpoints (Upload handling)
│   ├── core/config.py            # Global App Settings
│   ├── schemas/config.py         # Pydantic Models (Validation)
│   ├── services/
│   │   ├── data_service.py       # Data Cleaning & Splitting Logic
│   │   ├── ml_service.py         # Training & Testing Orchestration
│   │   └── model_factory.py      # Factory Pattern for ML Models
│   └── main.py                   # Entry Point
├── data/                         # CSV Files (Input/Output)
├── models/                       # Saved Models (.pkl, .json, .keras)
├── requirements.txt              # Dependencies
└── README.md                     # Documentation


## 🚀 Installation

1.  **Clone the repository:**

    git clone [https://github.com/BaranDpp/fastapi-ml-service.git](https://github.com/BaranDpp/fastapi-ml-service.git)
    cd fastapi-ml-service

2.  **Create & Activate Virtual Environment:**

    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate


3.  **Install Dependencies:**

    pip install -r requirements.txt

4.  **Run the Server:**

    uvicorn app.main:app --reload

    *Access Swagger UI at: [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)*


## 📖 Usage Guide (3-Step Pipeline)

### Step 0: Prepare Data

Place your raw CSV file (e.g., `my_data.csv`) inside the `data/` folder.

### Step 1: Data Processing (`/process-data`)

Cleans the data, handles missing values, and splits it into Train/Test sets.

**Example JSON Config:**

```json
{
  "raw_data_path": "data/my_data.csv",
  "target_column": "target",
  "feature_columns": ["temperature", "humidity", "soil_moisture", "rain", "pH"],
  "test_size": 0.2,
  "output_train_path": "data/processed_train.csv",
  "output_test_path": "data/processed_test.csv"
}
```

### Step 2: Model Training (`/train-model`)

Trains a model using the processed training data.

**Example JSON Config:**

```json
{
  "experiment_name": "experiment_01",
  "train_data_path": "data/processed_train.csv",
  "target_column": "target",
  "feature_columns": ["temperature", "humidity", "soil_moisture", "rain", "pH"],
  "algorithm_config": {
    "type": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    }
  },
  "save_model_path": "models/my_model.pkl"
}
```

*Supported types: `random_forest`, `xgboost`, `neural_network`.*

### Step 3: Model Testing (`/test-model`)

Evaluates the trained model against the test set and generates a report.

**Example JSON Config:**

```json
{
  "test_data_path": "data/processed_test.csv",
  "model_path": "models/my_model.pkl",
  "model_type": "random_forest",
  "target_column": "target",
  "feature_columns": ["temperature", "humidity", "soil_moisture", "rain", "pH"],
  "output_report_path": "reports/test_results.json"
}
```
 🤝 Contribution

1.  Fork the project.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit changes (`git commit -m 'feat: Add NewFeature'`).
4.  Push to branch & Open a Pull Request.

-----

Developed by **Baran**| Computer Engineer

```
```
