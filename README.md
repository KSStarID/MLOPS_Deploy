# Proyek Machine Learning Operations (MLOps)

## Dataset

Dataset yang digunakan adalah Apple Quality. Dataset ini berisi informasi tentang karakteristik apel seperti berat, manis, renyah, dan lain-lain yang digunakan untuk menentukan kualitas apel (baik atau buruk).

## Persoalan

Persoalan yang ingin diselesaikan adalah bagaimana mengklasifikasikan apel berdasarkan kualitasnya (baik atau buruk) menggunakan karakteristik fisik dan sensorik dari apel.

## Solusi Machine Learning

Solusi yang diterapkan adalah membuat model klasifikasi menggunakan TensorFlow yang akan memprediksi kualitas apel berdasarkan fitur-fitur yang ada. Target yang ingin dicapai adalah akurasi prediksi di atas 85%.

## Pengolahan Data dan Arsitektur Model

- **Pengolahan Data**: Data preprocessing dilakukan menggunakan komponen Transform dari TFX. Fitur numerik dinormalisasi menggunakan z-score normalization.
- **Arsitektur Model**: Model yang digunakan adalah Neural Network dengan 2 hidden layer.
- **Metrik Evaluasi**: Metrik yang digunakan adalah accuracy, precision, dan recall.

## Performa Model

Model yang dihasilkan mencapai akurasi sebesar 89% pada data validasi, dengan precision 87% dan recall 88%.

## Model Deployment

Model di-deploy menggunakan platform Railway dengan pendekatan containerization menggunakan Docker. API endpoint dibuat menggunakan TensorFlow Serving yang dikombinasikan dengan FastAPI untuk menyediakan antarmuka prediksi.

## Tautan Web App

URL untuk mengakses model serving: [https://apple-quality-model-kstarid.up.railway.app/](https://apple-quality-model-kstarid.up.railway.app/)

## Monitoring

Sistem monitoring menggunakan Prometheus untuk melacak performa model dan infrastruktur. Dashboard Grafana digunakan untuk visualisasi metrik seperti jumlah request, latency, dan akurasi prediksi.
