# 🚀 FastAPI ML Pipeline & Prediction Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📖 Proje Hakkında

Bu proje, ham veri setlerinin işlenmesi, makine öğrenmesi modellerinin dinamik olarak eğitilmesi ve test sonuçlarının raporlanması süreçlerini otomatize eden bir **RESTful API** çözümüdür.

Geleneksel statik scriptlerin aksine, bu sistem:
1.  **Esnek Veri Girişi:** Excel (.xlsx) ve CSV formatlarını destekler.
2.  **Dinamik Konfigürasyon:** Model parametreleri ve hedef değişkenler (target) kod değiştirilmeden JSON ile yönetilir.
3.  **End-to-End Akış:** Veri yüklemeden sonuç indirmeye kadar tüm süreç API üzerinden yönetilir.

*(Not: Ekran görüntülerindeki örnek veri seti, tarımsal sensör verileri ve gübre kullanım tahmini üzerine kurgulanmıştır.)*

## ✨ Temel Özellikler

* **📂 Çoklu Format Desteği:** `.csv` ve `.xlsx` dosyalarını otomatik algılar ve işler.
* **⚙️ Config-Driven Training:** Eğitim parametreleri (Epoch, Model Tipi, Feature Listesi) JSON üzerinden gönderilir.
* **📊 Otomatik Raporlama:** Test sonuçlarını ve tahminleri indirilebilir rapor haline getirir.
* **⚡ Yüksek Performans:** FastAPI ve Asenkron yapı sayesinde hızlı yanıt süreleri.

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel ortamınızda ayağa kaldırmak için aşağıdaki adımları izleyin.

### 1. Repoyu Klonlayın
git clone [https://github.com/kullaniciadi/proje-ismi.git](https://github.com/kullaniciadi/proje-ismi.git)
cd proje-ismi

2. Bağımlılıkları Yükleyin

pip install -r requirements.txt

4. Uygulamayı Başlatın
uvicorn app.main:app --reload
