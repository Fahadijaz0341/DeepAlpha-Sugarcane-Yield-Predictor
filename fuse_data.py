import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

def fuse_datasets():
    print("Loading datasets...")
    # 1. Load both datasets from the local workspace
    if not os.path.exists('sugarcane_data.csv') or not os.path.exists('DATASET.csv'):
        print("Error: Please ensure 'sugarcane_data.csv' and 'DATASET.csv' are in the same folder as this script.")
        return

    df_govt = pd.read_csv('sugarcane_data.csv')
    df_kaggle = pd.read_csv('DATASET.csv')

    print("Cleaning and standardizing yields...")
    # Drop rows where Govt yield is missing to ensure clean matching
    df_govt = df_govt.dropna(subset=['yield (mds/acre)']).copy()

    # 2. Convert Government Yield (Maunds/Acre) to Quintals/Acre to match Kaggle
    # 1 Maund = ~40 kg. 1 Quintal = 100 kg. Therefore: mds * 40 / 100 = quintals
    df_govt['yield_quintal_per_acre_govt'] = (df_govt['yield (mds/acre)'] * 40) / 100

    print("Running Nearest Neighbor Agronomic Matching...")
    # 3. K-Nearest Neighbor Matching
    govt_yields = df_govt[['yield_quintal_per_acre_govt']].values
    kaggle_yields = df_kaggle[['yield_quintal_per_acre']].values

    # Calculate the absolute difference between every Govt yield and Kaggle yield
    distances = cdist(govt_yields, kaggle_yields, metric='cityblock')

    # Find the index of the Kaggle row with the closest yield match for each Govt row
    closest_kaggle_indices = np.argmin(distances, axis=1)

    # 4. Fuse the Datasets
    # Create a new dataframe from the matched Kaggle rows
    matched_kaggle_data = df_kaggle.iloc[closest_kaggle_indices].reset_index(drop=True)

    # Reset Govt index to align properly before side-by-side merging
    df_govt = df_govt.reset_index(drop=True)

    # Combine them side-by-side
    fused_df = pd.concat([df_govt, matched_kaggle_data], axis=1)

    # Cleanup: Drop the redundant Kaggle yield column
    fused_df = fused_df.drop(columns=['yield_quintal_per_acre'])

    print("\n--- Fusion Complete! ---")
    print(f"Total Master Rows: {len(fused_df)}")
    
    # Save the ultimate tabular dataset
    output_filename = 'master_tabular_fused.csv'
    fused_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved as: {output_filename}")

if __name__ == "__main__":
    fuse_datasets()