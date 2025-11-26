# coding=utf-8
"""
crop_health_model_agentV6.py
CropHealthModelAgent using a saved Keras GRU model (V6.0).
STATUS: FULLY FIXED - All attribute errors resolved + proper forecast dates
Author: Samantha Lee
"""

import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fuzzywuzzy import process

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[3] 

MODEL_PATH = PROJECT_ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/saved_models/best_multivar_model_v6.0.keras"
SCALER_PATH = PROJECT_ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/utils/multivar_scaler_v6.0.pkl"

# Data Files
MERGED_DATA_FILE = PROJECT_ROOT / "data/final_lstm_dataset_cleaned.parquet" 
SET1_PATH = PROJECT_ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/training_results/SET1_growing_season_predictions.json"
SET2_PATH = PROJECT_ROOT / "machine_learning/src/feat/Crop_Health_Model_V6/training_results/SET2_non_growing_season_predictions.json"

SEQUENCE_LENGTH = 60
TARGET_COLUMN_NAME = 'NDVI_mean'

# --- GLOBAL DEFINITION OF VARIABLES ---
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

try:
    TARGET_COLUMN_INDEX = ALL_VARIABLE_COLS.index(TARGET_COLUMN_NAME)
except ValueError:
    TARGET_COLUMN_INDEX = 17  # Fallback to known position


class CropHealthModelAgentV6:
    name = "CropHealthModelAgentV6"

    def __init__(self, date: str = "2025-10-22", field_id: Optional[str] = None):
        # --- CRITICAL: Set these FIRST before anything else ---
        self.all_variable_cols = ALL_VARIABLE_COLS.copy()  # Use .copy() to ensure independence
        self.target_column_index = TARGET_COLUMN_INDEX
        
        self.model: Optional[keras.Model] = None
        self.scaler = None
        
        # DataFrames
        self.df_historical: Optional[pd.DataFrame] = None
        self.df_evaluation: Optional[pd.DataFrame] = None
        
        self.is_ready: bool = False
        self.using_eval_data: bool = False
        
        self.date = pd.to_datetime(date)
        self.field_id = field_id
        
        self.selected_row_data: Dict[str, Any] = {}
        self.location_map: Dict[str, str] = {} 
        self.all_locations: List[str] = [] 

        self._load_resources()
        if self.field_id:
            self.selected_row_data = self.get_data_by_date_field_id()

    def _load_resources(self):
        try:
            # 1. Load Main Parquet Data
            if MERGED_DATA_FILE.exists():
                self.df_historical = pd.read_parquet(MERGED_DATA_FILE, engine='fastparquet')
                self.df_historical["date"] = pd.to_datetime(self.df_historical["date"])
                self.df_historical = self.df_historical.sort_values(by=["field_id", "date"]).reset_index(drop=True)

            # 2. Load Evaluation JSONs
            eval_dfs = []
            for json_path in [SET1_PATH, SET2_PATH]:
                if json_path.exists():
                    try:
                        temp_df = pd.read_json(json_path)
                        if not temp_df.empty:
                            temp_df['date'] = pd.to_datetime(temp_df['date'])
                            # Standardize column names
                            if 'actual_ndvi' in temp_df.columns:
                                temp_df['NDVI_mean'] = temp_df['actual_ndvi']
                            eval_dfs.append(temp_df)
                    except Exception as e:
                        print(f"Warning: Could not load {json_path.name}: {e}")
            
            if eval_dfs:
                self.df_evaluation = pd.concat(eval_dfs, ignore_index=True)
                self.df_evaluation = self.df_evaluation.sort_values(by=["field_id", "date"]).reset_index(drop=True)

            self._build_location_map()

            # 3. Load Model
            if MODEL_PATH.exists() and SCALER_PATH.exists():
                self.model = keras.models.load_model(str(MODEL_PATH))
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.is_ready = True
            else:
                # Can still work with evaluation data
                self.is_ready = bool(self.df_evaluation is not None or self.df_historical is not None)

        except Exception as e:
            print(f"CRITICAL ERROR loading resources: {e}")
            import traceback
            traceback.print_exc()
            self.is_ready = False
            
    def _build_location_map(self):
        ids = []
        if self.df_historical is not None: 
            ids.extend(self.df_historical['field_id'].unique())
        if self.df_evaluation is not None: 
            ids.extend(self.df_evaluation['field_id'].unique())
        
        unique_ids = list(set(ids))
        temp_map = {}
        for fid in unique_ids:
            if '|' in fid:
                try:
                    parts = fid.split('|')
                    if len(parts) > 1:
                        original = parts[1]
                        clean = original.replace('-', ' ').title()
                        temp_map[clean] = original
                except: 
                    pass
        self.location_map = temp_map
        self.all_locations = sorted(list(self.location_map.keys()))

    @staticmethod
    def _interpret_health_status(ndvi_score: float) -> str:
        if ndvi_score > 0.6: return "Healthy"
        if 0.3 <= ndvi_score <= 0.6: return "Moderate"
        if 0.1 <= ndvi_score < 0.3: return "Stressed"
        return "Critical"

    def set_field_and_date(self, field_query: Optional[str], date: Optional[str], crop_query: Optional[str] = None):
        """Locates the specific Field ID based on Region and Crop."""
        
        if field_query:
            search_term = field_query
            best_match, score = process.extractOne(search_term, self.all_locations, score_cutoff=60)
            
            if not best_match:
                self.selected_row_data = {"error": f"Location '{field_query}' not found in available regions."}
                return

            original_region_str = self.location_map[best_match]
            
            # 1. Search Main Data
            match_df = pd.DataFrame()
            source = "main"
            
            if self.df_historical is not None:
                pattern = rf"\|{original_region_str}\|"
                match_df = self.df_historical[
                    self.df_historical['field_id'].str.contains(pattern, regex=True, case=False, na=False)
                ]

            # 2. Search Evaluation Data if needed
            if match_df.empty or crop_query:
                if self.df_evaluation is not None:
                    pattern = rf"\|{original_region_str}\|"
                    eval_match = self.df_evaluation[
                        self.df_evaluation['field_id'].str.contains(pattern, regex=True, case=False, na=False)
                    ]
                    
                    if not eval_match.empty:
                        use_eval = False
                        if match_df.empty: 
                            use_eval = True
                        elif crop_query:
                            crop_in_main = match_df['field_id'].str.contains(crop_query, case=False).any()
                            crop_in_eval = eval_match['field_id'].str.contains(crop_query, case=False).any()
                            if not crop_in_main and crop_in_eval: 
                                use_eval = True
                        
                        if use_eval:
                            match_df = eval_match
                            source = "eval"

            # 3. Apply Crop Filter
            if crop_query and not match_df.empty:
                crop_pattern = rf"\|{crop_query.lower()}"
                crop_df = match_df[match_df['field_id'].str.contains(crop_pattern, regex=True, case=False)]
                if not crop_df.empty: 
                    match_df = crop_df
            
            if not match_df.empty:
                self.field_id = match_df['field_id'].iloc[0]
                self.using_eval_data = (source == "eval")
            else:
                self.selected_row_data = {"error": f"No fields found for {field_query} (Crop: {crop_query})."}
                return

        if date: 
            self.date = pd.to_datetime(date)
            
        self.selected_row_data = self.get_data_by_date_field_id()

    def get_data_by_date_field_id(self) -> Dict[str, Any]:
        if self.field_id is None: 
            return {"error": "No field selected"}

        df_source = self.df_evaluation if self.using_eval_data else self.df_historical
        if df_source is None: 
            return {"error": "Data source unavailable"}

        mask = (df_source["date"] == self.date) & (df_source["field_id"] == self.field_id)
        filtered = df_source[mask]

        if filtered.empty:
            # Find nearest date
            field_data = df_source[df_source['field_id'] == self.field_id]
            if field_data.empty: 
                return {"error": "Field data not found."}
            
            nearest_idx = (field_data['date'] - self.date).abs().argsort()[:1]
            if len(nearest_idx) > 0:
                filtered = field_data.iloc[nearest_idx]
                self.date = filtered['date'].iloc[0]
            else:
                return {"error": "No matching dates found"}

        r = filtered.iloc[0]
        ndvi = round(float(r.get("NDVI_mean", r.get("actual_ndvi", 0))), 4)
        precip = round(float(r.get("weather_precip_sum", 0)), 2)
        temp = round(float(r.get("weather_temp_mean", 0)), 2)
        moist = round(float(r.get("smap_soil_moisture", 0)), 4)
        
        loc_name = "Unknown"
        crop_name = "Unknown"
        if '|' in self.field_id:
            parts = self.field_id.split('|')
            if len(parts) > 1: loc_name = parts[1].replace('-', ' ').title()
            if len(parts) > 2: crop_name = parts[2].capitalize()

        return {
            "date": self.date.strftime('%Y-%m-%d'),
            "field_id": self.field_id,
            "region": loc_name,
            "crop": crop_name,
            "ndvi_mean": ndvi,
            "health_status": self._interpret_health_status(ndvi),
            "precipitation_sum": precip,
            "temperature_mean": temp,
            "soil_moisture": moist,
            "source": "Evaluation Data" if self.using_eval_data else "Live Model"
        }

    def forecast_next_week(self) -> Dict[str, Any]:
        """
        Generates 7-day FUTURE forecast from anchor date.
        CRITICAL FIX: Always returns dates AFTER the anchor date.
        """
        if not self.is_ready:
            return {"error": "Model not initialized"}
        
        # --- PATH A: EVALUATION DATA ---
        if self.using_eval_data:
            if self.df_evaluation is None:
                return {"error": "Evaluation data not loaded"}
                
            field_df = self.df_evaluation[
                self.df_evaluation["field_id"] == self.field_id
            ].sort_values("date").reset_index(drop=True)
            
            if field_df.empty:
                return {"error": "No evaluation data for this field"}
            
            max_date = field_df['date'].max()
            
            # Check if we can forecast
            if self.date >= max_date:
                return {
                    "error": f"Cannot forecast beyond dataset end date ({max_date.strftime('%Y-%m-%d')}). Current anchor: {self.date.strftime('%Y-%m-%d')}"
                }

            forecast_list = []
            
            # CRITICAL: Start from day+1, not from current day
            for i in range(1, 8):
                target_date = self.date + timedelta(days=i)
                
                # Stop if we exceed data range
                if target_date > max_date:
                    break
                
                row = field_df[field_df["date"] == target_date]
                
                if not row.empty:
                    val = float(row.iloc[0].get("predicted_ndvi", 0))
                    status = self._interpret_health_status(val)
                    forecast_list.append({
                        "date": target_date.strftime("%Y-%m-%d"),
                        "predicted_ndvi_mean": round(val, 4),
                        "health_status": status
                    })
                else:
                    # Gap in data
                    break
            
            if not forecast_list:
                return {"error": "No forecast data available in the next 7 days"}

            return {
                "forecast": forecast_list,
                "current_conditions": self.selected_row_data,
                "anchor_date": self.date.strftime("%Y-%m-%d"),
                "note": "Pre-calculated predictions from evaluation dataset"
            }

        # --- PATH B: LIVE GRU MODEL ---
        if self.df_historical is None:
            return {"error": "Historical data not loaded"}
            
        field_df = self.df_historical[
            self.df_historical["field_id"] == self.field_id
        ].sort_values("date").reset_index(drop=True)
        
        if field_df.empty:
            return {"error": "No historical data for this field"}
        
        target_indices = field_df.index[field_df["date"] == self.date].tolist()
        
        if not target_indices: 
            return {"error": f"Anchor date {self.date.strftime('%Y-%m-%d')} not found in historical data"}
            
        idx = target_indices[0]
        
        if idx < SEQUENCE_LENGTH - 1: 
            return {"error": f"Need {SEQUENCE_LENGTH} days of history. Only {idx+1} available."}
        
        # Verify columns exist
        missing_cols = [col for col in self.all_variable_cols if col not in field_df.columns]
        if missing_cols:
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Extract sequence
        input_data = field_df.iloc[idx - SEQUENCE_LENGTH + 1 : idx + 1][self.all_variable_cols].values
        
        if self.scaler is None or self.model is None:
            return {"error": "Model or scaler not loaded"}
            
        input_scaled = self.scaler.transform(input_data)

        try:
            predictions = []
            current_seq = np.copy(input_scaled)
            
            # Generate 7 future predictions
            for _ in range(7):
                model_input = np.expand_dims(current_seq, axis=0)
                pred_vec = self.model.predict(model_input, verbose=0)[0]
                predictions.append(pred_vec)
                # Update sequence for next prediction
                current_seq = np.concatenate([current_seq[1:], np.expand_dims(pred_vec, axis=0)], axis=0)
            
            # Inverse transform
            predictions_orig = self.scaler.inverse_transform(np.array(predictions))
            ndvi_preds = predictions_orig[:, self.target_column_index]
            
            forecast_list = []
            # CRITICAL: Start from day+1
            for i, val in enumerate(ndvi_preds):
                f_date = self.date + timedelta(days=i+1)
                val = round(float(val), 4)
                forecast_list.append({
                    "date": f_date.strftime("%Y-%m-%d"),
                    "predicted_ndvi_mean": val,
                    "health_status": self._interpret_health_status(val)
                })
                
            return {
                "forecast": forecast_list,
                "current_conditions": self.selected_row_data,
                "anchor_date": self.date.strftime("%Y-%m-%d"),
                "note": "GRU model predictions"
            }
            
        except Exception as e:
            import traceback
            return {"error": f"Prediction failed: {str(e)}\n{traceback.format_exc()}"}