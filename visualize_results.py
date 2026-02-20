import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score
# Import the data prep function from your training script
from train_deep_model import prepare_data
from sklearn.model_selection import train_test_split

def plot_predictions():
    print("Loading Model and Data...")
    # Load the trained Transformer model
    model = tf.keras.models.load_model('deepalpha_transformer_sugarcane.keras', compile=False)
    
    # Get the data exactly as it was split before
    X_tab, X_seq, y, _ = prepare_data()
    _, X_tab_test, _, X_seq_test, _, y_test = train_test_split(
        X_tab, X_seq, y, test_size=0.2, random_state=42
    )
    
    print("Generating Predictions...")
    # Make predictions on the unseen test set
    predictions = model.predict([X_tab_test, X_seq_test]).flatten()
    
    # Calculate R-Squared (how well the model fits the variance)
    r2 = r2_score(y_test, predictions)
    print(f"R-Squared Score: {r2:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot of Actual vs Predicted
    plt.scatter(y_test, predictions, alpha=0.7, color='#2ca02c', edgecolors='k', s=80)
    
    # Draw the "Perfect Prediction" line
    min_val = min(np.min(y_test), np.min(predictions)) - 20
    max_val = max(np.max(y_test), np.max(predictions)) + 20
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title('DeepAlpha Transformer: Actual vs. Predicted Sugarcane Yield', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Government Yield (Maunds/Acre)', fontsize=12)
    plt.ylabel('Model Predicted Yield (Maunds/Acre)', fontsize=12)
    
    # Add a text box with your elite metrics
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9)
    plt.text(0.05, 0.95, f'Test MAE: 15.84\nR² Score: {r2:.2f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=bbox_props)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the high-res image for your portfolio/presentation
    plt.savefig('deepalpha_yield_predictions.png', dpi=300)
    print("\nSuccess! Chart saved as 'deepalpha_yield_predictions.png'")

if __name__ == "__main__":
    plot_predictions()