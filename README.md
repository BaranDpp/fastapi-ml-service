# 🚀 FastAPI Machine Learning Service

Bu proje, TensorFlow/Keras ile eğitilmiş bir Yapay Zeka modelini **FastAPI** kullanarak RESTful API servisine dönüştüren profesyonel bir backend uygulamasıdır.

Proje, **Clean Architecture** prensiplerine, **SOLID** kurallarına ve **Design Pattern**'lerine (Singleton, Service Layer) sadık kalınarak geliştirilmiştir.

## 🏗️ Mimari Yapı

Proje, kodun sürdürülebilirliğini ve test edilebilirliğini artırmak için katmanlı mimari (Layered Architecture) kullanır:

- **Router Layer:** HTTP isteklerini karşılar, yönlendirir.
- **Service Layer:** İş mantığını (Business Logic) ve ML operasyonlarını yönetir.
- **Schema Layer (DTO):** Veri doğrulama (Validation) ve tip güvenliğini sağlar.
- **Core:** Konfigürasyon ve ayarları yönetir.

## 🛠️ Teknoloji Yığını (Tech Stack)

- **Language:** Python 3.12+
- **Web Framework:** FastAPI
- **ML Engine:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Server:** Uvicorn
- **Validation:** Pydantic

## 📂 Proje Yapısı

fastapi_ml_project/
├── app/
│   ├── api/v1/         # Endpoint Controller'ları
│   ├── core/           # Config ve Ayarlar
│   ├── schemas/        # Pydantic Modelleri (Input/Output)
│   ├── services/       # AI Model Eğitimi ve Tahmin (Logic)
│   └── main.py         # Application Entry Point
├── data/               # Eğitim verileri (Gitignored)
├── models/             # Eğitilmiş .keras ve .pkl dosyaları (Gitignored)
├── requirements.txt    # Bağımlılıklar
└── README.md           # Dokümantasyon
🚀 Kurulum (Installation)
Projeyi yerel ortamınızda çalıştırmak için adımları izleyin:

Repoyu Klonlayın:
git clone [https://github.com/KULLANICI_ADIN/REPO_ISMI.git](https://github.com/KULLANICI_ADIN/REPO_ISMI.git)
cd REPO_ISMI
Sanal Ortamı (Venv) Oluşturun:
python -m venv venv
# Windows için aktivasyon:
.\venv\Scripts\activate
# Mac/Linux için:
source venv/bin/activate
Gerekli Kütüphaneleri Yükleyin:
pip install -r requirements.txt

🏃‍♂️ Çalıştırma (Usage)
Uygulamayı geliştirme modunda başlatmak için:
uvicorn app.main:app --reload
Sunucu https://www.google.com/search?q=http://127.0.0.1:8000 adresinde çalışmaya başlayacaktır.

📚 API Dokümantasyonu
Tarayıcınızda Swagger UI arayüzüne giderek API'yi test edebilirsiniz: 👉 https://www.google.com/search?q=http://127.0.0.1:8000/docs

Temel Endpointler:
POST /api/v1/train: Modeli verisetini kullanarak sıfırdan eğitir.

POST /api/v1/predict: Eğitilmiş modeli kullanarak tahmin yapar.
