# -*- coding: utf-8 -*-
"""
merge_datasets.py
Author: Samantha Lee
STATUS: UPDATED (Fixes Missing Zero-Shot Data)

Purpose:
1. Loads feature data (budapest_vas.parquet).
2. Loads coordinate data (CSVs).
3. Smarts Matches them to inject GPS into the main dataset.
4. CRITICAL: Merges 'zeroshot_ground_truth_with_coords.parquet' to add new regions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# --- 1. PATH CONFIGURATION ---
# Finds the 'data' folder relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 
DATA_DIR = PROJECT_ROOT / "data"

# Files
FEATURE_FILE = DATA_DIR / "budapest_vas.parquet"
CSV_FILES = {
    "Vas": DATA_DIR / "hu_2022_2025_sunflower_10d_fields_Vas.csv",
    "Budapest": DATA_DIR / "hu_2022_2025_sunflower_10d_fields_Budapest.csv"
}
ZEROSHOT_FILE = DATA_DIR / "zeroshot_ground_truth_with_coords.parquet"
OUTPUT_FILE = DATA_DIR / "final_lstm_dataset_cleaned.parquet"

def main():
    print(f"--- STARTING DATA PIPELINE ---")
    print(f"Data Directory: {DATA_DIR}")

    # --- STEP 1: LOAD MAIN FEATURES ---
    if not FEATURE_FILE.exists():
        print(f"CRITICAL ERROR: {FEATURE_FILE.name} not found.")
        return

    print(f"1. Loading Main Features ({FEATURE_FILE.name})...")
    df_features = pd.read_parquet(FEATURE_FILE)
    
    # Fingerprinting for match
    df_features['region_match'] = df_features['field_id'].apply(
        lambda x: 'Budapest' if 'Budapest' in str(x) else ('Vas' if 'Vas' in str(x) else 'Unknown')
    )
    df_features['ndvi_match'] = df_features['NDVI_mean'].round(4)

    # --- STEP 2: LOAD CSV COORDINATES ---
    print("2. Loading CSVs for GPS fix...")
    coord_frames = []
    for region, path in CSV_FILES.items():
        if path.exists():
            try:
                temp_df = pd.read_csv(path)
                rename_map = {'NDVI': 'NDVI_mean', 'lat': 'latitude', 'lon': 'longitude'}
                temp_df = temp_df.rename(columns={k:v for k,v in rename_map.items() if k in temp_df.columns})
                temp_df['date'] = pd.to_datetime(temp_df['date'])
                temp_df['region_match'] = region
                temp_df['ndvi_match'] = temp_df['NDVI_mean'].round(4)
                coord_frames.append(temp_df[['date', 'region_match', 'ndvi_match', 'latitude', 'longitude']])
            except Exception as e:
                print(f"   Error reading {region}: {e}")
        else:
            print(f"   WARNING: {path.name} missing!")

    if coord_frames:
        df_coords = pd.concat(coord_frames, ignore_index=True)
        
        # Smart Match
        match_df = pd.merge(
            df_features[['field_id', 'date', 'region_match', 'ndvi_match']],
            df_coords,
            on=['date', 'region_match', 'ndvi_match'],
            how='inner'
        )
        id_to_coords = match_df.groupby('field_id')[['latitude', 'longitude']].first().reset_index()
        print(f"   Mapped GPS for {len(id_to_coords)} Vas/Budapest fields.")
        
        # Inject
        df_final = pd.merge(df_features, id_to_coords, on='field_id', how='left')
    else:
        print("   Skipping CSV merge (No CSVs found). Using raw features.")
        df_final = df_features

    # --- STEP 3: FORMAT FIELD IDs ---
    print("3. Formatting IDs...")
    def update_id(row):
        if pd.isna(row.get('latitude')): return row['field_id']
        lat_code = int(row['latitude'] * 1000)
        lon_code = int(row['longitude'] * 1000)
        parts = str(row['field_id']).split('|')
        region = row['region_match']
        crop = parts[2] if len(parts) > 2 else 'sunflower'
        code = parts[3] if len(parts) > 3 else '0x0'
        return f"vtx|{region}|{crop}|{code}|+{lat_code}+{lon_code}"

    if 'latitude' in df_final.columns:
        df_final['field_id'] = df_final.apply(update_id, axis=1)

    # --- STEP 4: MERGE ZERO-SHOT DATA (CRITICAL FIX) ---
    print(f"4. Looking for Zero-Shot Data: {ZEROSHOT_FILE.name}")
    if ZEROSHOT_FILE.exists():
        try:
            df_zero = pd.read_parquet(ZEROSHOT_FILE)
            print(f"   Found file! Rows: {len(df_zero)}")
            
            # Standardize Columns (Handle different naming conventions)
            rename_map = {'lat': 'latitude', 'lon': 'longitude', 'NDVI': 'NDVI_mean'}
            df_zero = df_zero.rename(columns={k:v for k,v in rename_map.items() if k in df_zero.columns})
            
            # Ensure Date Format
            df_zero['date'] = pd.to_datetime(df_zero['date'])
            
            # Merge
            df_final = pd.concat([df_final, df_zero], ignore_index=True)
            print(f"   SUCCESS: Added {len(df_zero)} Zero-Shot rows.")
            
        except Exception as e:
            print(f"   ERROR reading Zero-Shot file: {e}")
    else:
        print("      WARNING: Zero-Shot file NOT found in 'data' folder.")
        print("      Please ensure 'zeroshot_ground_truth_with_coords.parquet' is uploaded.")

    # --- STEP 5: SAVE ---
    # Cleanup
    df_final = df_final.drop(columns=['region_match', 'ndvi_match'], errors='ignore')
    df_final = df_final.drop_duplicates(subset=['field_id', 'date'], keep='last')
    
    # Final Check
    total_rows = len(df_final)
    valid_gps = df_final['latitude'].notna().sum() if 'latitude' in df_final.columns else 0
    
    print(f"5. Saving to {OUTPUT_FILE.name}...")
    print(f"   Total Rows: {total_rows}")
    print(f"   Rows with GPS: {valid_gps}")
    
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print("--- DONE ---")

if __name__ == "__main__":
    main()