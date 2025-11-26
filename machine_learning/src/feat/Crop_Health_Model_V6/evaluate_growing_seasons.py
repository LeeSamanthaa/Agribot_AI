# coding=utf-8
"""
evaluate_growing_season.py
Author: Samantha Lee

UPDATED: Explicitly extracts latitude/longitude for JSON output.
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/saved_models/best_multivar_model_v6.0.keras"
SCALER_PATH = ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/utils/multivar_scaler_v6.0.pkl"
DATA_FILE = ROOT / "data/processed/crop_health/final_lstm_dataset_cleaned.parquet"
ORIGINAL_DATA_FILE = ROOT / "data/processed/crop_health/final_lstm_dataset_cleaned_GRUV6.parquet"
RESULTS_SAVE_PATH = Path("/home/slee213/Building-CropLogic_API/machine_learning/src/feat/Crop_Health_Model_V6/training_results")
PREDICTION_FILE_NAME = "SET1_growing_season_predictions.json"

SEQUENCE_LENGTH = 60
TARGET_COLUMN_NAME = 'NDVI_mean'
ALL_VARIABLE_COLS = [
    'weather_temp_mean', 'weather_temp_min', 'weather_temp_max',
    'weather_precip_sum', 'weather_humidity_mean', 'weather_wind_mean',
    'weather_pressure_mean', 'smap_soil_moisture', 'smap_surface_temp_C',
    'smap_veg_water', 'smap_clay_fraction', 'era5_2m_temperature',
    'era5_total_precipitation', 'era5_gdd_base5',
    'era5_volumetric_soil_water_layer_1', 'era5_sw_root',
    'era5_surface_solar_radiation_downwards', 'NDVI_mean', 'NDVI_min',
    'NDVI_max', 'day_of_year', 'month', 'week', 'era5_gdd_cumsum',
    'weather_precip_7d', 'weather_temp_7d_mean', 'is_growing_season',
    'heat_stress', 'cold_stress', 'drought_stress'
]
TARGET_COLUMN_INDEX = ALL_VARIABLE_COLS.index(TARGET_COLUMN_NAME)

SEASON_START = pd.to_datetime("2025-05-01")
SEASON_END = pd.to_datetime("2025-09-30")
FORECAST_DAYS = 7

def forecast_7_days(model, scaler, input_sequence):
    current_sequence = np.copy(input_sequence)
    forecasted_vectors_scaled = []
    for _ in range(FORECAST_DAYS):
        model_input = np.expand_dims(current_sequence, axis=0)
        predicted_vector = model.predict(model_input, verbose=0)[0]
        forecasted_vectors_scaled.append(predicted_vector)
        vector_to_append = np.expand_dims(predicted_vector, axis=0)
        current_sequence = np.concatenate([current_sequence[1:], vector_to_append], axis=0)
    
    forecast_orig = scaler.inverse_transform(np.array(forecasted_vectors_scaled))
    return forecast_orig[:, TARGET_COLUMN_INDEX]

def run_growing_season_evaluation():
    print("--- Starting SET 1 Evaluation (Growing Season) ---")
    
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        df = pd.read_parquet(DATA_FILE, engine='fastparquet')
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure coordinate columns exist
        if 'latitude' not in df.columns: df['latitude'] = np.nan
        if 'longitude' not in df.columns: df['longitude'] = np.nan
        
        df_original = pd.read_parquet(ORIGINAL_DATA_FILE, engine='fastparquet')
        original_fields = df_original['field_id'].unique()
        
    except Exception as e:
        print(f"\nCRITICAL ERROR loading assets: {e}")
        return

    new_fields_df = df[~df['field_id'].isin(original_fields)]
    unique_field_ids = new_fields_df['field_id'].unique()
    
    if len(unique_field_ids) == 0:
        print("\nERROR: No new fields found.")
        return

    print(f"\nFound {len(unique_field_ids)} new fields to evaluate.")
    
    results = []
    all_daily_data = [] 

    for field_id in unique_field_ids:
        field_df = new_fields_df[new_fields_df['field_id'] == field_id].sort_values(by='date')
        
        # Extract Region
        region_parts = field_id.split('|')
        if len(region_parts) > 1:
            region = region_parts[1].replace('-', ' ').title()
        else:
            region = "Unknown"
            
        # Extract Coords (Try columns first, then ID)
        lat = field_df['latitude'].iloc[0]
        lon = field_df['longitude'].iloc[0]
        
        if pd.isna(lat) and len(region_parts) > 4:
             # Fallback to parsing ID: ...|+47500+19500
             try:
                 coords_str = region_parts[-1] # "+47500+19500"
                 parts = coords_str.split('+')
                 lat = int(parts[1]) / 1000.0
                 lon = int(parts[2]) / 1000.0
             except:
                 pass

        all_predictions = []
        all_truths = []
        
        for base_date in pd.date_range(start=SEASON_START, end=SEASON_END, freq=f'{FORECAST_DAYS}D'):
            input_end_date = base_date
            input_start_date = base_date - pd.Timedelta(days=SEQUENCE_LENGTH - 1)
            truth_start_date = base_date + pd.Timedelta(days=1)
            truth_end_date = base_date + pd.Timedelta(days=FORECAST_DAYS)
            
            input_df = field_df[(field_df['date'] >= input_start_date) & (field_df['date'] <= input_end_date)]
            truth_df = field_df[(field_df['date'] >= truth_start_date) & (field_df['date'] <= truth_end_date)]
            
            if len(input_df) == SEQUENCE_LENGTH and len(truth_df) == FORECAST_DAYS:
                try:
                    input_data = input_df[ALL_VARIABLE_COLS].values.astype(np.float32)
                    scaled_input_data = scaler.transform(input_data)
                    predicted_ndvi = forecast_7_days(model, scaler, scaled_input_data)
                    ground_truth_ndvi = truth_df[TARGET_COLUMN_NAME].values
                    
                    all_predictions.extend(predicted_ndvi)
                    all_truths.extend(ground_truth_ndvi)
                    
                    daily_df = pd.DataFrame({
                        'date': truth_df['date'].dt.strftime('%Y-%m-%d').values, 
                        'field_id': field_id,
                        'region': region,
                        'latitude': lat,   # ADDED
                        'longitude': lon,  # ADDED
                        'base_date': base_date.strftime('%Y-%m-%d'), 
                        'predicted_ndvi': predicted_ndvi,
                        'actual_ndvi': ground_truth_ndvi
                    })
                    all_daily_data.append(daily_df)
                    
                except Exception as e:
                    pass

        if all_truths:
            mae = mean_absolute_error(all_truths, all_predictions)
            results.append({'region': region, 'field_id': field_id, 'mae': mae})
            print(f"   SUCCESS {region}: MAE = {mae:.4f}")

    if results:
        results_df = pd.DataFrame(results)
        print(f"   Average MAE: {results_df['mae'].mean():.4f}")
        
        if all_daily_data:
            final_daily_df = pd.concat(all_daily_data, ignore_index=True)
            save_path = RESULTS_SAVE_PATH / PREDICTION_FILE_NAME
            save_path.parent.mkdir(parents=True, exist_ok=True)
            final_daily_df.to_json(save_path, orient='records', date_format='iso') 
            print(f"\nSaved JSON with coordinates to: {save_path}")

if __name__ == "__main__":
    run_growing_season_evaluation()