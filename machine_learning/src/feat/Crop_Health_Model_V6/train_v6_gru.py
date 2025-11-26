#
# train_v6_gru.py
#
# This version is a robust, production-ready pipeline.
#
# V6.0 CHANGES:
#  - LEARNING_RATE: 0.0005 -> 0.0001 (To fix "Training is unstable" warning)
#  - PATIENCE: 15 -> 20 (To give the slower learning rate more time)
#
# FIXES & UPGRADES (from v3/v4):
#
# 1. Critical Data Pipeline Fix:
#    - Fixed Data Leakage: The `MinMaxScaler` is now `fit` *only* on
#      the training data. This was the main bug in the v3/v4 scripts.
#
# 2. Model Architecture:
#    - Converted Model: All `LSTM` layers were changed to `GRU` layers.
#
# 3. Diagnostics & Robustness:
#    - Added Data Quality Report: Runs `comprehensive_data_quality_report`
#      at the start to find missing data, duplicates, and small fields.
#    - Added Sequence Validation: `validate_sequence_generation` now
#      confirms sequence counts and checks for batch-size inefficiencies.
#    - Upgraded Training Analysis: Replaced simple plots with a 6-panel
#      `analyze_training_dynamics` plot (overfitting, stability, etc.).
#    - Added Advanced Residual Analysis: Added a 9-panel
#      `advanced_residual_analysis` plot (bias, normality, R-squared).
#    - Added R-squared (R2): R2 score is now a standard metric in all reports.
#
# 4. Reporting & Code Quality:
#    - Comprehensive JSON Summary: Saves `model_summary.json` with all
#      hyperparameters, data split date ranges, and performance metrics.
#    - Cleaned Log Output: Removed all emojis and replaced
#      with professional text logs (ERROR:, OK:, WARNING:).
#    - Fixed `plt.show()` Blocker: Removed all `plt.show()` calls.
#      The script now runs start-to-finish without user interaction.
#    - Fixed Bugs: Corrected the `total_sequences` NameError and removed
#      all duplicate functions from v5.1.
#    - Corrected BASE_PATH: Set to your '/mnt/c/...' path.
#    - v5.2 Fix: Fixed JSON serialization bug for `training_stable`.
#

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle
from pathlib import Path
import json
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

print(f"TensorFlow Version: {tf.__version__}")

# =============================================================================
# ## Step 1: Configuration
# =============================================================================

# --- Define the Absolute Project Path ---
BASE_PATH = Path('/mnt/c/Users/SCLee/OneDrive/Projects/crop_model')
SCRIPT_VERSION = "v6.0"

# Ensure this directory exists
BASE_PATH.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
INPUT_FILE = BASE_PATH / 'final_lstm_dataset_cleaned.parquet'

# --- Output Files ---
MODEL_FILE = BASE_PATH / f'best_multivar_model_{SCRIPT_VERSION}.keras'
SCALER_FILE = BASE_PATH / f'multivar_scaler_{SCRIPT_VERSION}.pkl'
SPLIT_FILE = BASE_PATH / f'gru_data_split_{SCRIPT_VERSION}.json'
SUMMARY_FILE = BASE_PATH / f'model_summary_{SCRIPT_VERSION}.json'

# --- Plotting Files ---
PLOT_FILE_7DAY_BEST = BASE_PATH / f'multivar_7day_forecast_BEST_{SCRIPT_VERSION}.png'
PLOT_FILE_7DAY_WORST = BASE_PATH / f'multivar_7day_forecast_WORST_{SCRIPT_VERSION}.png'
PLOT_FILE_TRAINING_DYNAMICS = BASE_PATH / f'training_dynamics_detailed_{SCRIPT_VERSION}.png'
PLOT_FILE_RESIDUAL_ANALYSIS = BASE_PATH / f'residual_analysis_{SCRIPT_VERSION}.png'

# --- Sequence Parameters ---
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 1
TARGET_COLUMN_NAME = 'NDVI_mean'

# --- Model Parameters ---
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0001  # <-- NEW: Reduced for stability
GRU_UNITS = [256, 128, 64]
DENSE_UNITS = 64
PATIENCE = 20           # <-- NEW: Increased for slower LR
DROPOUT_RATE = 0.2
 
# --- Splitting Parameters ---
TEST_SPLIT_SIZE = 0.15
VAL_SPLIT_SIZE = 0.15


# =============================================================================
# ## Step 2: Load and Define Variables
# =============================================================================

def load_raw_data():
    """Loads and preps the raw data (unscaled)."""
    print("\n" + "=" * 70)
    print(" STEP 2: LOADING RAW DATA")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found at {INPUT_FILE}")
        return None, None, -1, -1

    try:
        df = pd.read_parquet(INPUT_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['field_id', 'date']).reset_index(drop=True)

        print(f"OK: Loaded data shape: {df.shape}")
        print(f"OK: Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"OK: Number of unique fields: {df['field_id'].nunique()}")

        all_variable_cols = [col for col in df.columns if col not in ['field_id', 'date']]
        n_features = len(all_variable_cols)

        try:
            target_column_index = all_variable_cols.index(TARGET_COLUMN_NAME)
        except ValueError:
            print(f"ERROR: Target column '{TARGET_COLUMN_NAME}' not in data.")
            print(f"Available columns: {all_variable_cols[:10]}...")
            return None, None, -1, -1

        print(f"OK: Target column: {TARGET_COLUMN_NAME} (index {target_column_index})")
        print(f"OK: Total features: {n_features}")
        print("=" * 70 + "\n")

        return df, all_variable_cols, n_features, target_column_index

    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None, -1, -1


# =============================================================================
# ## Step 3: Data Quality Report
# =============================================================================

def comprehensive_data_quality_report(df_raw, all_variable_cols):
    """Comprehensive data quality checks."""
    print("\n" + "=" * 70)
    print(" DATA QUALITY REPORT")
    print("=" * 70)

    report = {
        'total_rows': len(df_raw),
        'total_fields': df_raw['field_id'].nunique(),
        'date_range': (str(df_raw['date'].min()), str(df_raw['date'].max())),
        'issues_found': []
    }

    # 1. Missing values
    print("\n1. MISSING VALUES CHECK")
    missing = df_raw[all_variable_cols].isnull().sum()
    if missing.sum() > 0:
        print("CRITICAL: Missing values found!")
        for col in missing[missing > 0].index:
            pct = (missing[col] / len(df_raw)) * 100
            print(f"   {col}: {missing[col]:,} ({pct:.2f}%)")
            report['issues_found'].append(f"Missing: {col} - {pct:.2f}%")
    else:
        print("OK: No missing values")

    # 2. Duplicates
    print("\n2. DUPLICATE CHECK")
    duplicates = df_raw.duplicated(subset=['field_id', 'date']).sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates:,} duplicate field_id + date combinations")
        report['issues_found'].append(f"Duplicates: {duplicates}")
    else:
        print("OK: No duplicates")

    # 3. Field sizes
    print("\n3. FIELD SIZE DISTRIBUTION")
    field_sizes = df_raw.groupby('field_id').size()
    print(f"   Mean observations per field: {field_sizes.mean():.0f}")
    print(f"   Median: {field_sizes.median():.0f}")
    print(f"   Min: {field_sizes.min()}, Max: {field_sizes.max()}")

    small_fields = (field_sizes < SEQUENCE_LENGTH + 7).sum()
    if small_fields > 0:
        pct_small = (small_fields / len(field_sizes)) * 100
        print(f"   WARNING: {small_fields} fields ({pct_small:.1f}%) too small for {SEQUENCE_LENGTH}-day + 7-day forecast")
        report['issues_found'].append(f"Small fields: {small_fields} ({pct_small:.1f}%)")

    # 4. Target variable
    print("\n4. TARGET VARIABLE ANALYSIS")
    if TARGET_COLUMN_NAME in df_raw.columns:
        print(f"   Range: [{df_raw[TARGET_COLUMN_NAME].min():.4f}, {df_raw[TARGET_COLUMN_NAME].max():.4f}]")
        print(f"   Mean: {df_raw[TARGET_COLUMN_NAME].mean():.4f}")
        print(f"   Std: {df_raw[TARGET_COLUMN_NAME].std():.4f}")
        print(f"   Skewness: {df_raw[TARGET_COLUMN_NAME].skew():.4f}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(report['issues_found'])} issue types found")
    print(f"{'=' * 70}\n")

    return report


# =============================================================================
# ## Step 4: Split Fields
# =============================================================================

def split_fields(unique_field_ids):
    """Splits the field IDs into train, val, and test sets."""
    print("\n" + "=" * 70)
    print(" STEP 3: SPLITTING FIELDS")
    print("=" * 70)

    try:
        test_fields, non_test_fields = train_test_split(
            unique_field_ids,
            test_size=(1 - TEST_SPLIT_SIZE),
            random_state=42
        )
        train_fields, val_fields = train_test_split(
            non_test_fields,
            test_size=(VAL_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)),
            random_state=42
        )

        print(f"OK: Training fields:   {len(train_fields):>6} ({len(train_fields) / len(unique_field_ids) * 100:.1f}%)")
        print(f"OK: Validation fields: {len(val_fields):>6} ({len(val_fields) / len(unique_field_ids) * 100:.1f}%)")
        print(f"OK: Test fields:       {len(test_fields):>6} ({len(test_fields) / len(unique_field_ids) * 100:.1f}%)")
        print("=" * 70 + "\n")

        return train_fields, val_fields, test_fields

    except Exception as e:
        print(f"ERROR splitting fields: {e}")
        return [], [], []


# =============================================================================
# ## Step 5: Sequence Validation
# =============================================================================

def validate_sequence_generation(field_data_cache, train_fields, val_fields, test_fields):
    """Validates sequence generation - FIXED VERSION."""
    print("\n" + "=" * 70)
    print(" SEQUENCE GENERATION VALIDATION")
    print("=" * 70)

    def count_sequences(field_list):
        total = 0
        for fid in field_list:
            field_len = len(field_data_cache[fid])
            seq_count = max(0, field_len - SEQUENCE_LENGTH - FORECAST_HORIZON + 1)
            total += seq_count
        return total

    n_train = count_sequences(train_fields)
    n_val = count_sequences(val_fields)
    n_test = count_sequences(test_fields)
    total_sequences = n_train + n_val + n_test  #  FIXED: Calculate BEFORE checking

    # NOW check if zero
    if total_sequences == 0:
        print("CRITICAL ERROR: Total sequences is zero!")
        print(f"   Check SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) vs data length")
        return {
            'train_sequences': 0, 'val_sequences': 0, 'test_sequences': 0,
            'total_sequences': 0, 'data_utilization_pct': 0
        }

    print(f"\nSequence Counts ({SEQUENCE_LENGTH}-day sequences):")
    print(f"   Training:   {n_train:>10,} ({n_train / total_sequences * 100:.1f}%)")
    print(f"   Validation: {n_val:>10,} ({n_val / total_sequences * 100:.1f}%)")
    print(f"   Test:       {n_test:>10,} ({n_test / total_sequences * 100:.1f}%)")
    print(f"   TOTAL:      {total_sequences:>10,}")

    total_rows = sum([len(field_data_cache[fid]) for fid in field_data_cache.keys()])
    print(f"\nOriginal rows: {total_rows:,}")
    print(f"Data utilization: {(total_sequences / total_rows) * 100:.1f}%")

    print(f"\nBatch Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {n_train // BATCH_SIZE:,}")
    print(f"   Validation steps: {n_val // BATCH_SIZE:,}")

    print("=" * 70 + "\n")

    return {
        'train_sequences': n_train,
        'val_sequences': n_val,
        'test_sequences': n_test,
        'total_sequences': total_sequences,
        'data_utilization_pct': float((total_sequences / total_rows) * 100) if total_rows > 0 else 0
    }


# =============================================================================
# ## Step 6: Create Data Generators
# =============================================================================

def sequence_generator(field_id_list, field_data_cache):
    """A Python generator that yields (X, y) sequences."""
    for field_id in field_id_list:
        field_data = field_data_cache[field_id]

        for j in range(len(field_data) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1):
            X_seq = field_data[j: j + SEQUENCE_LENGTH]
            y_target_vector = field_data[j + SEQUENCE_LENGTH + FORECAST_HORIZON - 1]
            yield (X_seq, y_target_vector)


def create_dataset(field_list, field_data_cache, n_features):
    """Creates a tf.data.Dataset from the sequence generator."""
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(field_list, field_data_cache),
        output_signature=(
            tf.TensorSpec(shape=(SEQUENCE_LENGTH, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(n_features,), dtype=tf.float32)
        )
    )
    dataset = dataset.shuffle(1024).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# ## Step 7: Create Test Set (In-Memory)
# =============================================================================

def get_test_set(test_fields, field_data_cache):
    """Creates the X_test, y_test arrays in memory."""
    print("\n" + "=" * 70)
    print(" STEP 6: CREATING TEST SET")
    print("=" * 70)

    X_set, y_set = [], []

    try:
        for x, y in sequence_generator(test_fields, field_data_cache):
            X_set.append(x)
            y_set.append(y)

        X_test = np.array(X_set)
        y_test = np.array(y_set)

        if X_test.shape[0] == 0:
            print("ERROR: No test sequences were generated. Check data and splits.")
            return None, None

        print(f"OK: X_test shape: {X_test.shape}")
        print(f"OK: y_test shape: {y_test.shape}")
        print(f"OK: Test sequences: {X_test.shape[0]:,}")
        print("=" * 70 + "\n")

        return X_test, y_test

    except Exception as e:
        print(f"ERROR creating test set: {e}")
        return None, None


# =============================================================================
# ## Step 8: Build Model
# =============================================================================

def build_model(n_features):
    """Builds the GRU model with Batch Normalization."""
    print("\n" + "=" * 70)
    print(f" STEP 7: BUILDING {SCRIPT_VERSION} GRU MODEL")
    print("=" * 70)

    model = keras.Sequential()
    model.add(layers.Input(shape=(SEQUENCE_LENGTH, n_features)))

    model.add(layers.GRU(GRU_UNITS[0], return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.GRU(GRU_UNITS[1], return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.GRU(GRU_UNITS[2], return_sequences=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(DENSE_UNITS, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(n_features))

    model.summary()
    print("=" * 70 + "\n")
    return model


# =============================================================================
# ## Step 9: Train Model
# =============================================================================

def train_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps):
    """Compiles and trains the model."""
    print("\n" + "=" * 70)
    print(" STEP 8: COMPILING AND TRAINING")
    print("=" * 70)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        str(MODEL_FILE), save_best_only=True, monitor='val_loss', verbose=1
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"OK: Training on GPU: {gpus[0].name}\n")
    else:
        print("WARNING: Training on CPU (this will be slower)\n")

    start_time = time.time()

    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[checkpoint_cb, early_stopping_cb],
            verbose=1
        )

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\nOK: Training complete in {training_time:.2f} seconds ({training_time / 60:.1f} minutes)")

        # Find best epoch metrics
        best_epoch_index = np.argmin(history.history['val_loss'])
        final_val_loss = history.history['val_loss'][best_epoch_index]
        final_val_mae = history.history['val_mean_absolute_error'][best_epoch_index]

        print(f"OK: Best epoch: {best_epoch_index + 1}")
        print(f"OK: Best val_loss: {final_val_loss:.6f}")
        print(f"OK: Best val_mae: {final_val_mae:.6f}")
        print("=" * 70 + "\n")

        return history, training_time, final_val_loss, final_val_mae

    except Exception as e:
        print(f"ERROR during training: {e}")
        return None, 0, -1, -1


# =============================================================================
# ## Step 10: Analyze Training Dynamics
# =============================================================================

def analyze_training_dynamics(history):
    """Analyzes training for overfitting."""
    print("\n" + "=" * 70)
    print(" TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)

    if history is None or 'loss' not in history.history:
        print("WARNING: History unavailable, skipping analysis")
        return {}

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])

    # Convergence metrics
    print("\n1. CONVERGENCE METRICS")
    train_reduction = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    val_reduction = (val_loss[0] - val_loss[-1]) / val_loss[0] * 100
    print(f"   Training loss reduction: {train_reduction:.1f}%")
    print(f"   Validation loss reduction: {val_reduction:.1f}%")

    best_epoch = np.argmin(val_loss)
    print(f"   Best validation at epoch: {best_epoch + 1}")

    # Overfitting metrics
    print("\n2. OVERFITTING METRICS")
    best_gap = val_loss[best_epoch] - train_loss[best_epoch]
    best_gap_pct = (best_gap / train_loss[best_epoch]) * 100

    print(f"   Best epoch train/val gap: {best_gap:.6f} ({best_gap_pct:.1f}%)")

    # Diagnosis
    if best_gap_pct < 5:
        print(f"   OK (EXCELLENT): Minimal overfitting (<5%)")
        diagnosis = "excellent"
    elif best_gap_pct < 15:
        print(f"   OK (GOOD): Acceptable overfitting (5-15%)")
        diagnosis = "good"
    elif best_gap_pct < 30:
        print(f"   WARNING (MODERATE): Some overfitting (15-30%)")
        diagnosis = "moderate"
    else:
        print(f"   SEVERE: Significant overfitting (>30%)")
        diagnosis = "severe"

    # Training stability
    print("\n3. TRAINING STABILITY")
    val_loss_diff = np.diff(val_loss)
    sign_changes = np.sum(np.diff(np.sign(val_loss_diff)) != 0)
    print(f"   Validation loss oscillations: {sign_changes}")

    if sign_changes > len(val_loss) * 0.5:
        print(f"   WARNING: Training is unstable")
    else:
        print(f"   OK: Training is stable")

    # Create visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Dynamics Analysis ({SCRIPT_VERSION})', fontsize=16, fontweight='bold')

        # Plot 1: Loss curves
        axes[0, 0].plot(train_loss, label='Train', linewidth=2)
        axes[0, 0].plot(val_loss, label='Val', linewidth=2)
        axes[0, 0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label='Best')
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Log scale
        axes[0, 1].semilogy(train_loss, label='Train', linewidth=2)
        axes[0, 1].semilogy(val_loss, label='Val', linewidth=2)
        axes[0, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Loss (Log Scale)', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (MSE, log)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Gap
        gap = val_loss - train_loss
        axes[0, 2].plot(gap, linewidth=2, color='red')
        axes[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].fill_between(range(len(gap)), 0, gap, alpha=0.3, color='red')
        axes[0, 2].set_title('Overfitting Gap', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Gap (MSE)')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: MAE if available
        if 'mean_absolute_error' in history.history:
            axes[1, 0].plot(history.history['mean_absolute_error'], label='Train MAE', linewidth=2)
            axes[1, 0].plot(history.history['val_mean_absolute_error'], label='Val MAE', linewidth=2)
            axes[1, 0].set_title('MAE Curves', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'MAE not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)

        # Plot 5: Reduction rate
        train_reduction_rate = -np.diff(train_loss) / train_loss[:-1] * 100
        val_reduction_rate = -np.diff(val_loss) / val_loss[:-1] * 100
        axes[1, 1].plot(train_reduction_rate, label='Train', linewidth=2)
        axes[1, 1].plot(val_reduction_rate, label='Val', linewidth=2)
        axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Loss Reduction Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reduction (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Ratio
        ratio = val_loss / train_loss
        axes[1, 2].plot(ratio, linewidth=2, color='purple')
        axes[1, 2].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect')
        axes[1, 2].fill_between(range(len(ratio)), 1.0, ratio, alpha=0.3, color='purple')
        axes[1, 2].set_title('Val/Train Ratio', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Ratio')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(PLOT_FILE_TRAINING_DYNAMICS, dpi=150)
        plt.close(fig)  # CRITICAL: Close the figure
        print("\nOK: Training dynamics plot saved")
    except Exception as e:
        print(f"\nWARNING: Could not create training plot: {e}")

    print("=" * 70 + "\n")

    return {
        'best_epoch': int(best_epoch + 1),
        'best_epoch_gap_percent': float(best_gap_pct),
        'train_reduction_percent': float(train_reduction),
        'val_reduction_percent': float(val_reduction),
        'training_stable': bool(sign_changes < len(val_loss) * 0.5), # <-- THE FIX IS HERE
        'overfitting_diagnosis': diagnosis
    }


# =============================================================================
# ## Step 11: Evaluate Model
# =============================================================================

def evaluate_model(X_test, y_test, scaler, target_column_index):
    """Loads the best model, evaluates it, and returns predictions."""
    print("\n" + "=" * 70)
    print(" STEP 9: EVALUATING ON TEST SET")
    print("=" * 70)

    print(f"Loading best model from {MODEL_FILE}...")

    try:
        best_model = keras.models.load_model(str(MODEL_FILE))
        print("OK: Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None, None, None, -1, -1, -1, -1

    try:
        test_loss_scaled, test_mae_scaled = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"OK: Test Loss (MSE, scaled): {test_loss_scaled:.6f}")
        print(f"OK: Test MAE (scaled): {test_mae_scaled:.6f}")

        print("Making predictions on test set...")
        y_pred_scaled = best_model.predict(X_test, verbose=0)

        if y_pred_scaled.shape[1] != y_test.shape[1]:
            print(f"ERROR: Shape mismatch: {y_pred_scaled.shape} vs {y_test.shape}")
            return None, None, None, -1, -1, -1, -1

        y_pred_orig = scaler.inverse_transform(y_pred_scaled)
        y_test_orig = scaler.inverse_transform(y_test)

        # Evaluate target column
        ndvi_test_orig = y_test_orig[:, target_column_index]
        ndvi_pred_orig = y_pred_orig[:, target_column_index]

        rmse_final = np.sqrt(mean_squared_error(ndvi_test_orig, ndvi_pred_orig))
        mae_final = mean_absolute_error(ndvi_test_orig, ndvi_pred_orig)

        print(f"\nOK: Final {TARGET_COLUMN_NAME} Metrics (1-day forecast):")
        print(f"   RMSE: {rmse_final:.4f}")
        print(f"   MAE:  {mae_final:.4f}")
        print("=" * 70 + "\n")

        return best_model, y_test_orig, y_pred_orig, rmse_final, mae_final, test_loss_scaled, test_mae_scaled

    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        return None, None, None, -1, -1, -1, -1


# =============================================================================
# ## Step 12: Advanced Residual Analysis
# =============================================================================

def advanced_residual_analysis(y_test_orig, y_pred_orig, target_idx, all_variable_cols):
    """Deep residual analysis on unscaled data."""
    print("\n" + "=" * 70)
    print(" ADVANCED RESIDUAL ANALYSIS")
    print("=" * 70)

    y_true_ndvi = y_test_orig[:, target_idx]
    y_pred_ndvi = y_pred_orig[:, target_idx]

    errors = y_true_ndvi - y_pred_ndvi
    abs_errors = np.abs(errors)

    print("\n1. ERROR STATISTICS (Unscaled)")
    print(f"   Mean Error (Bias):       {errors.mean():.6f}")
    print(f"   Mean Absolute Error:     {abs_errors.mean():.6f}")
    print(f"   RMSE:                    {np.sqrt((errors ** 2).mean()):.6f}")
    print(f"   R-squared (R2):          {r2_score(y_true_ndvi, y_pred_ndvi):.6f}")
    print(f"   Median Absolute Error:   {np.median(abs_errors):.6f}")
    print(f"   95th Percentile Error:   {np.percentile(abs_errors, 95):.6f}")
    print(f"   Max Error:               {abs_errors.max():.6f}")

    # Bias check
    if abs(errors.mean()) > 0.01:
        if errors.mean() > 0:
            print(f"   WARNING: Model UNDER-predicts on average")
        else:
            print(f"   WARNING: Model OVER-predicts on average")
    else:
        print(f"   OK: No systematic bias")

    print("\n2. ERROR DISTRIBUTION")
    print(f"   Skewness: {pd.Series(errors).skew():.4f}")
    print(f"   Kurtosis: {pd.Series(errors).kurt():.4f}")

    if abs(pd.Series(errors).skew()) < 0.5:
        print(f"   OK: Errors are approximately normal")
    else:
        print(f"   WARNING: Errors are skewed")

    print("\n3. PERFORMANCE BY VALUE RANGE")
    quartiles = np.percentile(y_true_ndvi, [0, 25, 50, 75, 100])
    for i in range(len(quartiles) - 1):
        mask = (y_true_ndvi >= quartiles[i]) & (y_true_ndvi < quartiles[i + 1])
        if mask.sum() > 0:
            range_mae = abs_errors[mask].mean()
            print(f"   [{quartiles[i]:.3f}, {quartiles[i + 1]:.3f}): MAE = {range_mae:.6f}")

    # Create visualization
    try:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        fig.suptitle(f'Residual Analysis ({SCRIPT_VERSION})', fontsize=16, fontweight='bold')

        # Plot 1: Residuals vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_pred_ndvi, errors, alpha=0.1, s=1)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted NDVI')
        ax1.set_ylabel('Residual')
        ax1.set_title('Residuals vs Predicted', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals vs True
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(y_true_ndvi, errors, alpha=0.1, s=1)
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('True NDVI')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residuals vs True', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(errors, bins=100, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {errors.mean():.4f}')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Q-Q plot
        ax4 = fig.add_subplot(gs[1, 0])
        stats.probplot(errors, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Absolute error
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(y_pred_ndvi, abs_errors, alpha=0.1, s=1)
        ax5.set_xlabel('Predicted NDVI')
        ax5.set_ylabel('Absolute Error')
        ax5.set_title('Absolute Error vs Predicted', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Error percentiles
        ax6 = fig.add_subplot(gs[1, 2])
        percentiles = np.arange(0, 101, 5)
        error_percentiles = np.percentile(abs_errors, percentiles)
        ax6.plot(percentiles, error_percentiles, linewidth=2)
        ax6.axhline(abs_errors.mean(), color='red', linestyle='--',
                    label=f'Mean: {abs_errors.mean():.4f}')
        ax6.set_xlabel('Percentile')
        ax6.set_ylabel('Absolute Error')
        ax6.set_title('Error Percentile Curve', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Error by bin
        ax7 = fig.add_subplot(gs[2, 0])
        bins = np.linspace(y_true_ndvi.min(), y_true_ndvi.max(), 20)
        bin_indices = np.digitize(y_true_ndvi, bins)
        bin_maes = [abs_errors[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0
                    for i in range(1, len(bins))]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax7.bar(bin_centers, bin_maes, width=(bins[1] - bins[0]) * 0.8, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('True NDVI')
        ax7.set_ylabel('MAE')
        ax7.set_title('MAE by Value Range', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Scatter colored by error
        ax8 = fig.add_subplot(gs[2, 1])
        scatter = ax8.scatter(y_true_ndvi, y_pred_ndvi, c=abs_errors,
                              alpha=0.3, s=1, cmap='YlOrRd')
        ax8.plot([y_true_ndvi.min(), y_true_ndvi.max()],
                 [y_true_ndvi.min(), y_true_ndvi.max()],
                 'b--', linewidth=2, label='Perfect')
        ax8.set_xlabel('True NDVI')
        ax8.set_ylabel('Predicted NDVI')
        ax8.set_title('Predictions by Error', fontweight='bold')
        plt.colorbar(scatter, ax=ax8, label='Abs Error')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Plot 9: Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        summary_text = f"""
ERROR SUMMARY
═════════════════════
Mean Error:   {errors.mean():.6f}
MAE:          {abs_errors.mean():.6f}
RMSE:         {np.sqrt((errors ** 2).mean()):.6f}
R2:           {r2_score(y_true_ndvi, y_pred_ndvi):.6f}

Median AE:    {np.median(abs_errors):.6f}
95th %ile:    {np.percentile(abs_errors, 95):.6f}
Max Error:    {abs_errors.max():.6f}

Skewness:     {pd.Series(errors).skew():.4f}
Kurtosis:     {pd.Series(errors).kurt():.4f}
        """
        ax9.text(0.05, 0.5, summary_text, transform=ax9.transAxes,
                 fontsize=10, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(PLOT_FILE_RESIDUAL_ANALYSIS, dpi=150)
        plt.close(fig)  # CRITICAL: Close the figure
        print("\nOK: Residual analysis plot saved")
    except Exception as e:
        print(f"\nWARNING: Could not create residual plot: {e}")

    print("=" * 70 + "\n")

    return {
        'mean_error': float(errors.mean()),
        'mae': float(abs_errors.mean()),
        'rmse': float(np.sqrt((errors ** 2).mean())),
        'r2_score': float(r2_score(y_true_ndvi, y_pred_ndvi)),
        'median_ae': float(np.median(abs_errors)),
        'p95_error': float(np.percentile(abs_errors, 95))
    }


# =============================================================================
# ## Step 13: Visualize 7-Day Forecasts
# =============================================================================

def visualize_forecast(best_model, scaler, test_fields, field_data_cache, target_column_index, df_raw):
    """Finds and plots the best AND worst 7-day forecasts."""
    print("\n" + "=" * 70)
    print(" STEP 10: VISUALIZING 7-DAY FORECASTS")
    print("=" * 70)

    print("Searching for best/worst forecasts (checking 20 fields)...")
    best_field_id, worst_field_id = None, None
    best_field_mae, worst_field_mae = np.inf, -np.inf
    best_field_plot_data, worst_field_plot_data = {}, {}

    fields_to_check = test_fields[:min(20, len(test_fields))]

    for field_id in fields_to_check:
        try:
            field_data = field_data_cache[field_id]
            if len(field_data) < SEQUENCE_LENGTH + 7:
                continue

            starting_sequence = field_data[0:SEQUENCE_LENGTH]
            actual_vectors = field_data[SEQUENCE_LENGTH: SEQUENCE_LENGTH + 7]
            actual_orig = scaler.inverse_transform(actual_vectors)
            actual_7day_ndvi = actual_orig[:, target_column_index]

            forecast_orig = forecast_7_days(best_model, scaler, starting_sequence)
            forecast_7day_ndvi = forecast_orig[:, target_column_index]

            if len(actual_7day_ndvi) < 7:
                continue

            this_mae = mean_absolute_error(actual_7day_ndvi, forecast_7day_ndvi)
            plot_dates = df_raw[df_raw['field_id'] == field_id]['date'].iloc[SEQUENCE_LENGTH: SEQUENCE_LENGTH + 7]

            current_plot_data = {
                'dates': plot_dates,
                'actual': actual_7day_ndvi,
                'forecast': forecast_7day_ndvi
            }

            if this_mae < best_field_mae:
                best_field_mae, best_field_id, best_field_plot_data = this_mae, field_id, current_plot_data

            if this_mae > worst_field_mae:
                worst_field_mae, worst_field_id, worst_field_plot_data = this_mae, field_id, current_plot_data

        except Exception as e:
            continue

    # Plot BEST Case
    if best_field_id is not None:
        print(f"OK: Plotting BEST forecast (MAE: {best_field_mae:.4f}) - Field: {best_field_id}")
        try:
            fig = plt.figure(figsize=(15, 6))
            plt.plot(best_field_plot_data['dates'], best_field_plot_data['actual'],
                     label='Actual NDVI', color='blue', marker='o', linewidth=2)
            plt.plot(best_field_plot_data['dates'], best_field_plot_data['forecast'],
                     label='7-Day Forecast', color='green', linestyle='--', marker='o', linewidth=2)
            plt.title(f'BEST 7-Day NDVI Forecast (Field: {best_field_id}, MAE: {best_field_mae:.4f})',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Date'), plt.ylabel('NDVI_mean'), plt.legend(), plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_FILE_7DAY_BEST, dpi=150)
            plt.close(fig) # CRITICAL: Close the figure
        except Exception as e:
            print(f"WARNING: Could not plot best field: {e}")

    # Plot WORST Case
    if worst_field_id is not None:
        print(f"OK: Plotting WORST forecast (MAE: {worst_field_mae:.4f}) - Field: {worst_field_id}")
        try:
            fig = plt.figure(figsize=(15, 6))
            plt.plot(worst_field_plot_data['dates'], worst_field_plot_data['actual'],
                     label='Actual NDVI', color='blue', marker='o', linewidth=2)
            plt.plot(worst_field_plot_data['dates'], worst_field_plot_data['forecast'],
                     label='7-Day Forecast', color='red', linestyle='--', marker='o', linewidth=2)
            plt.title(f'WORST 7-Day NDVI Forecast (Field: {worst_field_id}, MAE: {worst_field_mae:.4f})',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Date'), plt.ylabel('NDVI_mean'), plt.legend(), plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_FILE_7DAY_WORST, dpi=150)
            plt.close(fig) # CRITICAL: Close the figure
        except Exception as e:
            print(f"WARNING: Could not plot worst field: {e}")

    print("=" * 70 + "\n")


# =============================================================================
# ## MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(f" STARTING SCRIPT: {SCRIPT_VERSION}")
    print("=" * 70)

    # Initialize summary
    model_summary = {
        "date_ranges": {},
        "models": {
            "metadata": {
                "report_generated_at_utc": str(datetime.utcnow()),
                "script_version": SCRIPT_VERSION,
                "tensorflow_version": tf.__version__
            }
        }
    }

    # 1. Load data
    df_raw, all_variable_cols, n_features, target_idx = load_raw_data()

    if df_raw is None:
        print("FATAL: Could not load data. Exiting.")
        exit(1)

    # 2. Data quality report
    quality_report = comprehensive_data_quality_report(df_raw, all_variable_cols)
    model_summary['data_quality_report'] = quality_report

    # 3. Split fields
    unique_field_ids = list(df_raw['field_id'].unique())
    train_fields, val_fields, test_fields = split_fields(unique_field_ids)

    if len(train_fields) == 0:
        print("FATAL: No training fields. Exiting.")
        exit(1)

    # 4. Calculate date ranges
    df_train_only = df_raw[df_raw['field_id'].isin(train_fields)]
    df_val_only = df_raw[df_raw['field_id'].isin(val_fields)]
    df_test_only = df_raw[df_raw['field_id'].isin(test_fields)]

    model_summary["date_ranges"]["train"] = {
        "start": str(df_train_only['date'].min().date()),
        "end": str(df_train_only['date'].max().date()),
        "rows": len(df_train_only)
    }
    model_summary["date_ranges"]["val"] = {
        "start": str(df_val_only['date'].min().date()),
        "end": str(df_val_only['date'].max().date()),
        "rows": len(df_val_only)
    }
    model_summary["date_ranges"]["test"] = {
        "start": str(df_test_only['date'].min().date()),
        "end": str(df_test_only['date'].max().date()),
        "rows": len(df_test_only)
    }

    # 5. Save split
    try:
        with open(SPLIT_FILE, 'w') as f:
            json.dump({
                'train_fields': train_fields,
                'val_fields': val_fields,
                'test_fields': test_fields
            }, f, indent=4)
        print(f"OK: Split file saved: {SPLIT_FILE}")
    except Exception as e:
        print(f"WARNING: Could not save split file: {e}")

    # 6. Fit scaler on training data ONLY
    print("\n" + "=" * 70)
    print(" STEP 4: FITTING SCALER (Training Data Only)")
    print("=" * 70)
    main_scaler = MinMaxScaler(feature_range=(0, 1))
    main_scaler.fit(df_train_only[all_variable_cols])

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(main_scaler, f)
    print(f"OK: Scaler saved: {SCALER_FILE}")
    print("=" * 70 + "\n")

    # 7. Transform all data
    df_scaled = df_raw.copy()
    df_scaled[all_variable_cols] = main_scaler.transform(df_raw[all_variable_cols])

    # 8. Create data cache
    print("Grouping data by field...")
    data_cache = {
        field_id: field_df[all_variable_cols].values
        for field_id, field_df in df_scaled.groupby('field_id')
    }
    print(f"OK: Cached {len(data_cache)} fields\n")

    # 9. Validate sequences
    sequence_info = validate_sequence_generation(data_cache, train_fields, val_fields, test_fields)
    model_summary['models']['data_sequences'] = sequence_info

    if sequence_info['total_sequences'] == 0:
        print("FATAL: No sequences generated. Exiting.")
        exit(1)

    # 10. Create datasets
    print("Creating TensorFlow datasets...")
    train_ds = create_dataset(train_fields, data_cache, n_features)
    val_ds = create_dataset(val_fields, data_cache, n_features)
    print("OK: Datasets created\n")

    # 11. Create test set
    X_test, y_test = get_test_set(test_fields, data_cache)

    if X_test is None:
        print("FATAL: Could not create test set. Exiting.")
        exit(1)

    steps_per_epoch = sequence_info['train_sequences'] // BATCH_SIZE
    validation_steps = sequence_info['val_sequences'] // BATCH_SIZE

    # 12. Build model
    model = build_model(n_features)

    # Populate summary
    model_summary['models']['model_version'] = SCRIPT_VERSION
    model_summary['models']['architecture'] = "GRU"
    model_summary['models']['n_features'] = n_features
    model_summary['models']['target_column'] = TARGET_COLUMN_NAME
    model_summary['models']['parameters'] = {
        'sequence_length': SEQUENCE_LENGTH,
        'forecast_horizon': FORECAST_HORIZON,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'gru_units': GRU_UNITS,
        'dense_units': DENSE_UNITS,
        'patience': PATIENCE,
        'dropout_rate': DROPOUT_RATE
    }

    # 13. Train model
    history, training_time, val_loss, val_mae = train_model(
        model, train_ds, val_ds, steps_per_epoch, validation_steps
    )

    if history is None:
        print("FATAL: Training failed. Exiting.")
        exit(1)

    model_summary['models']['training_time_seconds'] = round(training_time, 2)
    model_summary['models']['best_validation_metrics'] = {
        'val_loss_mse': round(val_loss, 6),
        'val_mae': round(val_mae, 6)
    }

    # 14. Analyze training dynamics
    training_analysis = analyze_training_dynamics(history)
    model_summary['models']['training_dynamics'] = training_analysis

    # 15. Evaluate model
    (best_model, y_test_orig, y_pred_orig,
     rmse, mae, test_loss_scaled, test_mae_scaled) = evaluate_model(
        X_test, y_test, main_scaler, target_idx
    )

    if best_model is None:
        print("FATAL: Evaluation failed. Exiting.")
        exit(1)

    model_summary['models']['test_metrics_scaled'] = {
        'mse': round(test_loss_scaled, 6),
        'mae': round(test_mae_scaled, 6)
    }

    # 16. Residual analysis
    residual_report = advanced_residual_analysis(
        y_test_orig, y_pred_orig, target_idx, all_variable_cols
    )
    model_summary['models']['test_metrics_unscaled'] = residual_report

    # 17. Visualize forecasts
    visualize_forecast(best_model, main_scaler, test_fields, data_cache, target_idx, df_raw)

    # 18. Save summary
    print(f"Saving model summary to {SUMMARY_FILE}...")
    try:
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(model_summary, f, indent=4)
        print(f"OK: Summary saved successfully")
    except Exception as e:
        print(f"WARNING: Error saving summary: {e}")

    # Final report
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print(f"Model Version:        {SCRIPT_VERSION}")
    print(f"Training Time:        {training_time / 60:.1f} minutes")
    print(f"Best Val Loss:        {val_loss:.6f}")
    print(f"Test RMSE (unscaled): {rmse:.4f}")
    print(f"Test MAE (unscaled):  {mae:.4f}")
    print(f"R-squared (R2):       {residual_report['r2_score']:.4f}")

    if 'overfitting_diagnosis' in training_analysis:
        print(f"Overfitting Status:   {training_analysis['overfitting_diagnosis'].upper()}")
        print(f"                      ({training_analysis['best_epoch_gap_percent']:.1f}% gap)")

    print("\nOK: All outputs saved to:")
    print(f"  - Model: {MODEL_FILE}")
    print(f"  - Summary: {SUMMARY_FILE}")
    print(f"  - Plots: {BASE_PATH}")
    print("=" * 70)
    print("\nOK: SCRIPT COMPLETE\n")