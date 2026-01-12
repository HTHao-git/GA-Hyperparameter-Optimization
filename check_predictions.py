# ============================================================================
# CHECK PREDICTION DISTRIBUTION
# ============================================================================
# Verify if models are just predicting majority class
# ============================================================================

import numpy as np
from pathlib import Path
from preprocessing.data_loader import DatasetLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.colors import print_header, print_info, print_warning
import json

# Load best configs
output_dir = Path('outputs/model_comparison')

with open(output_dir / 'random_forest_results.json') as f:
    rf_results = json.load(f)

with open(output_dir / 'xgboost_results.json') as f:
    xgb_results = json.load(f)

# Load dataset
loader = DatasetLoader(interactive=False)
X, y, _ = loader.load_dataset('secom')

# Split same way as comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_header("PREDICTION ANALYSIS")
print()

print_info(f"Test set size: {len(y_test)}")
print_info(f"Test set distribution:")
print(f"  Class 0: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"  Class 1: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
print()

# Check what models would predict (simplified - no preprocessing)
print_info("If we predicted ALL class 0:")
all_zero_accuracy = np.sum(y_test == 0) / len(y_test)
print(f"  Accuracy: {all_zero_accuracy:.4f}")
print()

print_info("Actual model scores:")
rf_score = rf_results.get('test_score', rf_results.get('test_accuracy', 0))
xgb_score = xgb_results.get('test_accuracy', xgb_results.get('test_score', 0))

print(f"  Random Forest:    {rf_score:.4f}")
print(f"  XGBoost:         {xgb_score:.4f}")
print()

if abs(rf_score - all_zero_accuracy) < 0.01:
    print_warning("⚠️  Models are performing close to 'predict all majority class' baseline!")
    print_warning("   This suggests heavy class imbalance bias.")
    print()
    print_info("Recommendations:")
    print("  1. Check confusion matrix (are minority class predictions correct?)")
    print("  2. Look at precision/recall for each class")
    print("  3. Try different SMOTE strategies or class weights")
    print("  4. Consider F1-score or balanced accuracy as fitness metric")
else:
    print_info("✓ Models are beating the majority class baseline")

print()
print_info("Check the HTML report for confusion matrices!")