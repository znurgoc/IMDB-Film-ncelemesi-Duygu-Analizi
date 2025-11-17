# 🎬 IMDB Film İncelemesi Duygu Analizi (SimpleRNN ve KerasTuner)

Bu proje, **derin öğrenme** teknikleri kullanılarak IMDB film incelemelerinin duygusal tonunu (pozitif/negatif) sınıflandırmayı amaçlayan bir çalışmadır. Modelin hiperparametre optimizasyonu için **KerasTuner** kullanılmıştır.

---

## 📚 Proje Kaynağı ve Eğitim Bağlamı

Bu proje, bir eğitim/kurs kapsamında gerçekleştirilmiş uygulamalı bir çalışmadır.



Bu çalışma, öğrenilen temel RNN (Tekrarlayan Sinir Ağı) mimarisini uygulamayı ve model performansını otomatik olarak optimize etmeyi (Hyperparameter Tuning) öğrenme amacını taşımaktadır.

---

## ✨ Temel Özellikler ve Kullanılan Teknolojiler

Bu projede aşağıdaki teknolojiler ve yöntemler kullanılmıştır:

* **Veri Seti:** IMDB Duygu Analizi Veri Seti (Keras API üzerinden yüklenmiştir).
* **Model Mimarisi:** **SimpleRNN** (Basit Tekrarlayan Sinir Ağı) kullanılmıştır.
* **Ön İşleme:**
    * Veri seti, en sık kullanılan 10.000 kelime ile sınırlandırılmıştır.
    * Tüm incelemeler, **`maxlen=100`** olacak şekilde aynı uzunluğa getirilmiştir (**Padding**).
* **Hiperparametre Optimizasyonu:**
    * **KerasTuner** (`RandomSearch` metodu) kullanılarak en iyi model mimarisi bulunmuştur.
    * Optimize Edilen Parametreler: Embedding katmanı çıktı boyutu, RNN birim sayısı ve Dropout oranı.
* **Geri Çağrı (Callback):** **Erken Durdurma (Early Stopping)** kullanılarak aşırı öğrenmenin (Overfitting) önüne geçilmiştir.

---

## 🚀 Sonuçlar

Yapılan hiperparametre araması sonucunda elde edilen en iyi modelin test verileri üzerindeki performansı:

* **Test Doğruluğu (Accuracy):** `<0.830 >`
* **Test AUC:** `<0.910>`
* **ROC Eğrisi:** Model, pozitif ve negatif sınıfları ayırmada **çok başarılı** (AUC > 0.90) olduğunu göstermiştir.

<img width="771" height="586" alt="Ekran görüntüsü 2025-11-17 221207" src="https://github.com/user-attachments/assets/8deef5f8-7aa8-4975-af19-ccb0db003b0d" />

---

```bash
pip install numpy matplotlib tensorflow scikit-learn
pip install keras-tuner
