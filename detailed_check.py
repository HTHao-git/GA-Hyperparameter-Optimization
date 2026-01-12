# ============================================================================
# DETAILED PREDICTION CHECK
# ============================================================================

import numpy as np
import json
from pathlib import Path
from preprocessing.data_loader import DatasetLoader
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler
from preprocessing.smote_handler import SMOTEHandler
from preprocessing.pca import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils.colors import print_header, print_info, print_warning, print_success

print_header("DETAILED PREDICTION ANALYSIS")
print()

# Load data
loader = DatasetLoader(interactive=False)
X, y, _ = loader.load_dataset('secom')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_info(f"Test set:  {len(y_test)} samples")
print(f"  Class 0: {np.sum(y_test == 0)} (pass)")
print(f"  Class 1: {np.sum(y_test == 1)} (fail)")
print()

# Load best RF config
with open('outputs/model_comparison/random_forest_results.json') as f:
    rf_results = json.load(f)

# Debug:  see what keys are available
print_info("Available keys in RF results:")
print(f"  {list(rf_results.keys())}")
print()

# Get config (handle different possible key names)
if 'config' in rf_results:
    best_config = rf_results['config']
elif 'best_config' in rf_results: 
    best_config = rf_results['best_config']
else:
    print_warning("No config found in results, using defaults")
    best_config = {
        'n_estimators':  200,
        'max_depth':  15,
        'min_samples_split': 10,
        'min_samples_leaf': 8,
        'max_features':  'sqrt'
    }

print_info("Training Random Forest with best config (simplified)...")

# Simple preprocessing
mv_handler = MissingValuesHandler(strategy='mean')
X_train_clean = mv_handler.fit_transform(X_train)
X_test_clean = mv_handler.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

# Train simple RF
rf = RandomForestClassifier(
    n_estimators=best_config.get('n_estimators', 200),
    max_depth=best_config.get('max_depth'),
    min_samples_split=best_config.get('min_samples_split', 2),
    min_samples_leaf=best_config.get('min_samples_leaf', 1),
    max_features=best_config.get('max_features', 'sqrt'),
    random_state=42
)

rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print()
print_info("RESULTS:")
print()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted")
print(f"              Class 0  Class 1")
print(f"Actual Class 0    {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Class 1    {cm[1,0]:3d}     {cm[1,1]:3d}")
print()

# What percentage of class 1 was detected?
class_1_detected = cm[1,1]
class_1_total = np.sum(y_test == 1)
detection_rate = (class_1_detected / class_1_total * 100) if class_1_total > 0 else 0

print_info(f"Class 1 (minority) detection rate: {class_1_detected}/{class_1_total} ({detection_rate:.1f}%)")
print()

if detection_rate < 10: 
    print_warning("⚠️  Model is barely detecting minority class!")
    print_warning("   It's mostly predicting class 0 (majority)")
elif detection_rate < 50:
    print_warning("⚠️  Model detects some minority samples, but not well")
else:
    print_success("✓ Model is detecting minority class reasonably")

print()
print("Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0 (Pass)', 'Class 1 (Fail)']))

print()
print_info("Prediction distribution:")
print(f"  Predicted class 0: {np.sum(y_pred == 0)} ({np.sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
print(f"  Predicted class 1: {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.1f}%)")