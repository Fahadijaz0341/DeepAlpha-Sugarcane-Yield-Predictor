import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout,
                                     BatchNormalization, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepAlpha Yield Predictor", layout="wide", page_icon="🌾")

# --- MODEL ARCHITECTURE (must match train_deep_model.py) ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_model(tabular_shape=16, seq_length=30):
    tabular_input = Input(shape=(tabular_shape,), name="Tabular_Input")
    x1 = Dense(128, activation='relu')(tabular_input)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(64, activation='relu')(x1)

    seq_input = Input(shape=(seq_length, 1), name="NDVI_Sequence_Input")
    x2 = transformer_encoder(seq_input, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dense(64, activation='relu')(x2)

    merged = Concatenate()([x1, x2])
    z = Dense(128, activation='relu')(merged)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, name="Yield_Prediction")(z)

    return Model(inputs=[tabular_input, seq_input], outputs=output)

# --- LOAD MODEL & DATA (Cached) ---
@st.cache_resource
def load_deep_model():
    model = build_model(tabular_shape=16, seq_length=30)
    model.load_weights('model_weights.weights.h5')
    return model

@st.cache_data
def load_and_prep_data():
    df_tabular = pd.read_csv('master_tabular_fused.csv')
    df_ndvi = pd.read_csv('sentinel2_ndvi_5years.csv')

    num_cols = ['rainfall_mm', 'avg_temperature_celsius', 'sunlight_hours_per_day',
                'nitrogen_n_kg_per_acre', 'phosphorus_p_kg_per_acre', 'potassium_k_kg_per_acre',
                'crop_duration_days', 'irrigation_frequency_per_month', 'pest_control_applied']
    cat_cols = ['soil_type', 'seed_variety']

    scaler = StandardScaler()
    scaler.fit(df_tabular[num_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df_tabular[cat_cols])

    df_ndvi['Date'] = pd.to_datetime(df_ndvi['Date'])
    def map_year(date):
        year = date.year
        month = date.month
        if month >= 5: return f"{year}-{str(year+1)[-2:]}"
        else: return f"{year-1}-{str(year)[-2:]}"
    df_ndvi['match_year'] = df_ndvi['Date'].apply(map_year)
    df_ndvi['match_district'] = df_ndvi['District'].str.lower().str.strip()

    ndvi_dict = {}
    for (dist, yr), group in df_ndvi.groupby(['match_district', 'match_year']):
        seq = group.sort_values('Date')['Calculated_NDVI'].values.tolist()
        ndvi_dict[f"{dist.title()} ({yr})"] = seq

    return scaler, encoder, ndvi_dict

model = load_deep_model()
scaler, encoder, ndvi_dict = load_and_prep_data()

# --- APP UI ---
st.title("🌾 DeepAlpha: Multi-Modal Sugarcane Yield Predictor")
st.markdown("""
This application uses a **Transformer-based Deep Learning Dual-Input Architecture**. 
It fuses **tabular agronomic data** (weather, soil, fertilizers) with **Sentinel-2 Satellite time-series data** (NDVI) to predict crop yields in Punjab, Pakistan.
""")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Environmental & Soil Data")

    st.subheader("Weather Conditions")
    rain = st.slider("Rainfall (mm)", 500, 2500, 1200)
    temp = st.slider("Avg Temperature (°C)", 20.0, 45.0, 31.0)
    sun = st.slider("Sunlight (Hours/Day)", 4.0, 12.0, 8.0)

    st.subheader("Agronomic Inputs")
    soil = st.selectbox("Soil Type", ['Alluvial', 'Sandy Loam', 'Clay Loam', 'Red Soil'])
    seed = st.selectbox("Seed Variety", ['Co86032', 'Co0238', 'CoM0265'])
    n_kg = st.number_input("Nitrogen (kg/acre)", 50, 300, 150)
    p_kg = st.number_input("Phosphorus (kg/acre)", 20, 150, 60)
    k_kg = st.number_input("Potassium (kg/acre)", 20, 150, 70)

    dur = st.slider("Crop Duration (Days)", 250, 450, 360)
    irr = st.slider("Irrigation Frequency (/month)", 1, 8, 3)
    pest = st.selectbox("Pest Control Applied", [1, 0])

with col2:
    st.header("2. Satellite NDVI Time-Series")
    st.markdown("Select a historical satellite footprint to feed into the Transformer's Attention Mechanism.")

    available_seqs = list(ndvi_dict.keys())
    selected_seq_key = st.selectbox("Select District & Year (Sentinel-2 Data)", available_seqs)

    selected_sequence = ndvi_dict[selected_seq_key]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(selected_sequence, marker='o', color='green', linestyle='-', linewidth=2)
    ax.set_title(f"NDVI Profile: {selected_seq_key}", fontsize=10)
    ax.set_xlabel("Time Steps (Growing Season)", fontsize=8)
    ax.set_ylabel("NDVI", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.divider()

    if st.button("🚀 Run Deep Fusion Prediction", use_container_width=True):
        with st.spinner("Processing through Transformer and Dense layers..."):
            input_num = pd.DataFrame([[rain, temp, sun, n_kg, p_kg, k_kg, dur, irr, pest]],
                                     columns=['rainfall_mm', 'avg_temperature_celsius', 'sunlight_hours_per_day',
                                              'nitrogen_n_kg_per_acre', 'phosphorus_p_kg_per_acre', 'potassium_k_kg_per_acre',
                                              'crop_duration_days', 'irrigation_frequency_per_month', 'pest_control_applied'])
            input_cat = pd.DataFrame([[soil, seed]], columns=['soil_type', 'seed_variety'])

            scaled_num = scaler.transform(input_num)
            encoded_cat = encoder.transform(input_cat)
            X_tab_input = np.hstack([scaled_num, encoded_cat])

            MAX_SEQ_LENGTH = 30
            X_seq_padded = pad_sequences([selected_sequence], maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')
            X_seq_input = np.expand_dims(X_seq_padded, axis=-1)

            prediction = model.predict([X_tab_input, X_seq_input])[0][0]

            st.success("Analysis Complete!")
            st.metric(label="Predicted Sugarcane Yield", value=f"{prediction:.2f} Maunds/Acre", delta="High Confidence")
            st.caption("Note: 1 Maund = ~40kg")