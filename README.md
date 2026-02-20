# 🌾 DeepAlpha: Multi-Modal Sugarcane Yield Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)

## 📌 Project Overview
DeepAlpha is a state-of-the-art Multi-Modal Deep Learning framework designed to predict sugarcane yields in Punjab, Pakistan. Moving beyond standard tabular forecasting, this project implements a **Dual-Input Neural Network** that fuses micro-level agronomic data with macro-level remote sensing computer vision.

By utilizing a **Transformer-based Self-Attention Mechanism**, the model actively tracks the historical health of the crop over a 5-year growing season to predict the final harvest output with a **Mean Absolute Error (MAE) of just 15.84 Maunds/Acre** (~2% error margin).

## 🧠 Architecture: Deep Fusion
The framework utilizes a two-branch neural network:
1. **The Vision Branch (Transformer Encoder):** Ingests sequential NDVI (Normalized Difference Vegetation Index) time-series data pulled from **Copernicus Sentinel-2** satellites. A multi-head attention mechanism learns long-term temporal dependencies in crop health.
2. **The Agronomic Branch (Dense DNN):** Ingests environmental tabular data (soil type, rainfall, nitrogen levels, seed variety) using K-Nearest Neighbor imputed datasets from official government statistics.
3. **Fusion Layer:** Concatenates the embedded representations from both branches to output a highly accurate, continuous yield prediction.

## 📊 Data Engineering Pipeline
* **Automated PDF Parsing:** Extracted unstructured 5-year historical crop yields (Kharif Estimates) from government PDFs using `pdfplumber`.
* **Agronomic Imputation:** Fused official district-level data with Kaggle agronomic datasets using `scipy` spatial distance matching.
* **Earth Engine Integration:** Automated extraction of historical satellite footprints using the `earthengine-api` to calculate max/mean NDVI values per district.

## 🚀 Live Web Application
This repository includes a fully interactive **Streamlit** web dashboard. Users can adjust weather sliders, alter soil compositions, select historical satellite footprints, and run the Transformer model in real-time.
https://deepalpha-sugarcane-yield-predictor.streamlit.app/
