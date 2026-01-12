# ============================================================================
# TEST SMOTE WITH REAL DATA (SECOM)
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler
from preprocessing.smote_handler import SMOTEHandler
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success
from utils.colors import print_header, print_section, print_success, print_info
import numpy as np

logger = get_logger(name="SMOTE_TEST", verbose=True)

print_header("SMOTE TEST - SECOM DATASET")

# Load SECOM
loader = DatasetLoader(logger=logger, interactive=False)
X, y, metadata = loader.load_dataset('secom')

logger.blank()

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

print_section("Preprocessing Pipeline")

# Step 1: Handle missing values
logger.info("Step 1: Handling missing values...")
mv_handler = MissingValuesHandler(strategy='mean', logger=logger)
X_clean = mv_handler.fit_transform(X)

logger.blank()

# Step 2: Scale features
logger.info("Step 2: Scaling features...")
scaler = StandardScaler(logger=logger)
X_scaled = scaler.fit_transform(X_clean)

logger.blank()

# ============================================================================
# SMOTE RESAMPLING
# ============================================================================

print_section("SMOTE Resampling")

# Show original imbalance
smote_handler = SMOTEHandler(strategy='smote', k_neighbors=5, random_state=42, logger=logger)
smote_handler.report_imbalance(y)

logger.blank()

# Test different strategies
strategies = ['smote', 'random_over', 'random_under', 'combined']

results = {}

for strategy in strategies:
    logger.section(f"Strategy: {strategy.upper()}")
    
    handler = SMOTEHandler(strategy=strategy, k_neighbors=5, random_state=42, logger=logger)
    X_resampled, y_resampled = handler.fit_resample(X_scaled, y)
    
    results[strategy] = {
        'X':  X_resampled,
        'y': y_resampled,
        'shape': X_resampled.shape
    }
    
    logger.blank()

# ============================================================================
# SUMMARY
# ============================================================================

print_header("RESAMPLING SUMMARY")

print_info("Results:")
print()

for strategy, result in results.items():
    print(f"  {strategy:15} - Shape: {result['shape']} - Samples: {len(result['y'])}")

logger.blank()
print_success("✓ All resampling strategies work with SECOM!")
logger.blank()

print_info("Original SECOM:")
print(f"  Shape: {X.shape}")
print(f"  Class 0 (Fail): {(y == 0).sum()} samples")
print(f"  Class 1 (Pass): {(y == 1).sum()} samples")
print(f"  Imbalance:  {(y == 1).sum() / (y == 0).sum():.1f}:1")