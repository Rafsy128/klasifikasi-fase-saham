# klasifikasi-fase-saham
Klasifikasi fase pasar saham menggunakan Random Forest dan Support Vector Machine berbasis Price-Volume Action.
# Klasifikasi Fase Pasar Saham

## Deskripsi

Proyek ini merupakan implementasi klasifikasi fase pasar saham berdasarkan
pendekatan Price-Volume Action dan teori Wyckoff menggunakan algoritma
Random Forest dan Support Vector Machine (SVM).

Fase pasar yang diklasifikasikan terdiri dari:

- Accumulation
- Markup
- Distribution
- Markdown

## Tujuan

Membangun sistem yang dapat membantu mengidentifikasi fase pergerakan
saham berdasarkan data historis harga dan volume.

## Dataset

Dataset menggunakan data historis saham yang diperoleh dari Yahoo Finance.

Atribut utama:

- Date
- Open
- High
- Low
- Close
- Volume

## Feature Engineering

Fitur yang digunakan:

- Return
- RVOL
- TrendDiff
- Price Position
- Breakout
- Breakdown

## Metodologi

Alur penelitian:

Data Collection
→ Data Preprocessing
→ Exploratory Data Analysis
→ Feature Engineering
→ Rule-Based Labeling
→ Train-Test Split
→ Random Forest & SVM
→ Evaluation
→ Streamlit Dashboard

## Algoritma

### Random Forest

Random Forest digunakan sebagai model klasifikasi berbasis ensemble
yang terdiri dari beberapa decision tree.

### Support Vector Machine

SVM digunakan sebagai pembanding terhadap Random Forest dengan
normalisasi fitur menggunakan StandardScaler.

## Evaluasi

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Confusion Matrix

## Aplikasi

Dashboard dibuat menggunakan Streamlit untuk menampilkan:

- Dataset
- EDA
- Feature Engineering
- Rule-Based Labeling
- Train-Test Split
- Random Forest
- SVM
- Perbandingan Model
- Hasil Prediksi

## Instalasi

Clone repository:

git clone https://github.com/USERNAME/klasifikasi-fase-pasar-saham.git

Masuk ke folder:

cd klasifikasi-fase-pasar-saham

Install library:

pip install -r requirements.txt

## Menjalankan aplikasi

streamlit run app.py

## Teknologi

- Python
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Matplotlib
- Plotly
- Streamlit
- Joblib

## Author

Muhamad Rafiansyah

## License

Untuk keperluan akademik.
