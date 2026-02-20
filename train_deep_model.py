import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def prepare_data():
    print("Loading Multi-Modal Datasets...")
    df_tabular = pd.read_csv('master_tabular_fused.csv')
    df_ndvi = pd.read_csv('sentinel2_ndvi_5years.csv')

    # --- 1. PREPROCESS NDVI (TIME-SERIES) ---
    print("Processing Satellite Time-Series...")
    df_ndvi['Date'] = pd.to_datetime(df_ndvi['Date'])
    
    def map_year(date):
        year = date.year
        month = date.month
        if month >= 5: return f"{year}-{str(year+1)[-2:]}"
        else: return f"{year-1}-{str(year)[-2:]}"
        
    df_ndvi['match_year'] = df_ndvi['Date'].apply(map_year)
    df_ndvi['match_district'] = df_ndvi['District'].str.lower().str.strip()
    df_tabular['match_district'] = df_tabular['district'].str.lower().str.strip()

    ndvi_sequences = {}
    for (dist, yr), group in df_ndvi.groupby(['match_district', 'match_year']):
        seq = group.sort_values('Date')['Calculated_NDVI'].values.tolist()
        ndvi_sequences[f"{dist}_{yr}"] = seq

    # --- 2. PREPROCESS TABULAR DATA ---
    print("Processing Agronomic Tabular Data...")
    num_cols = ['rainfall_mm', 'avg_temperature_celsius', 'sunlight_hours_per_day', 
                'nitrogen_n_kg_per_acre', 'phosphorus_p_kg_per_acre', 'potassium_k_kg_per_acre',
                'crop_duration_days', 'irrigation_frequency_per_month', 'pest_control_applied']
    cat_cols = ['soil_type', 'seed_variety']
    
    df_tabular = df_tabular.dropna(subset=['yield (mds/acre)'])
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_tabular[num_cols])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df_tabular[cat_cols])
    
    X_tabular = np.hstack([X_num, X_cat])
    
    # --- 3. ALIGN AND FUSE DATA ---
    print("Aligning NDVI sequences with Tabular targets...")
    X_seq_list, X_tab_list, y_list = [], [], []
    
    for idx, row in df_tabular.iterrows():
        key = f"{row['match_district']}_{row['year']}"
        seq = ndvi_sequences.get(key, [0.0]) 
        
        X_seq_list.append(seq)
        X_tab_list.append(X_tabular[idx])
        y_list.append(row['yield (mds/acre)'])

    MAX_SEQ_LENGTH = 30
    X_seq_padded = pad_sequences(X_seq_list, maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')
    X_seq_padded = np.expand_dims(X_seq_padded, axis=-1)
    
    return np.array(X_tab_list), X_seq_padded, np.array(y_list), MAX_SEQ_LENGTH

# --- THE UPGRADE: TRANSFORMER ENCODER BLOCK ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Self-Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_fusion_model(tabular_shape, seq_length):
    # Branch 1: Tabular Agronomic Data
    tabular_input = Input(shape=(tabular_shape,), name="Tabular_Input")
    x1 = Dense(128, activation='relu')(tabular_input)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(64, activation='relu')(x1)
    
    # Branch 2: NDVI Time-Series (TRANSFORMER)
    seq_input = Input(shape=(seq_length, 1), name="NDVI_Sequence_Input")
    
    # Apply Transformer Block
    x2 = transformer_encoder(seq_input, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x2 = GlobalAveragePooling1D()(x2) # Compress the attention sequence into a single vector
    x2 = Dense(64, activation='relu')(x2)
    
    # Fusion: Concatenate the learned features
    merged = Concatenate()([x1, x2])
    
    # Output Branch
    z = Dense(128, activation='relu')(merged)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, name="Yield_Prediction")(z) 
    
    model = Model(inputs=[tabular_input, seq_input], outputs=output)
    
    # Using a learning rate scheduler for the complex Transformer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model

if __name__ == "__main__":
    X_tab, X_seq, y, seq_length = prepare_data()
    
    X_tab_train, X_tab_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
        X_tab, X_seq, y, test_size=0.2, random_state=42
    )
    
    model = build_transformer_fusion_model(tabular_shape=X_tab.shape[1], seq_length=seq_length)
    model.summary()
    
    print("\n--- Initiating Multi-Modal TRANSFORMER Training ---")
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        [X_tab_train, X_seq_train], y_train,
        validation_data=([X_tab_test, X_seq_test], y_test),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    
    loss, mae = model.evaluate([X_tab_test, X_seq_test], y_test)
    print(f"\nTransformer Model Evaluation -> Mean Absolute Error: {mae:.2f} Maunds/Acre")
    
    model.save('deepalpha_transformer_sugarcane.keras')
    print("SOTA Industry model successfully saved as 'deepalpha_transformer_sugarcane.keras'")