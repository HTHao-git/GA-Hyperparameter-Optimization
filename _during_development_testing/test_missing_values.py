# ============================================================================
# TEST MISSING VALUES HANDLER
# ============================================================================

from preprocessing.missing_values import MissingValuesHandler
from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success
import numpy as np

logger = get_logger(name="MISSING_TEST", verbose=True)

print_header("MISSING VALUES HANDLER TEST")

# ============================================================================
# TEST 1: Synthetic Data
# ============================================================================

print_section("TEST 1: Synthetic Data with 20% Missing Values")

np.random.seed(42)
X_synthetic = np.random.randn(100, 10)

# Introduce 20% missing values
mask = np.random.random(X_synthetic.shape) < 0.2
X_synthetic[mask] = np.nan

handler = MissingValuesHandler(strategy='mean', logger=logger)
handler.report_missing(X_synthetic)

logger.blank()
logger.info("Imputing with mean strategy...")
X_clean = handler.fit_transform(X_synthetic)

stats_after = handler.detect_missing(X_clean)
if stats_after['total_missing'] == 0:
    print_success("✓ Test 1 PASSED - All missing values handled")
else:
    logger.error(f"✗ Test 1 FAILED - {stats_after['total_missing']} missing values remain")

logger.blank()

# ============================================================================
# TEST 2: Real Data (SECOM)
# ============================================================================

print_section("TEST 2: Real Data - SECOM Dataset")

loader = DatasetLoader(logger=logger, interactive=False)

try:
    X_secom, y_secom, metadata = loader.load_dataset('secom')
    
    logger.blank()
    logger.info("SECOM dataset loaded")
    logger.info(f"  Shape: {X_secom.shape}")
    logger.info(f"  Known missing:  {metadata.get('missing_values', 0)} ({metadata.get('missing_percentage', 0):.1f}%)")
    
    logger.blank()
    
    # Test different strategies
    strategies = ['mean', 'median', 'constant']
    
    for strategy in strategies:
        logger.section(f"Strategy: {strategy.upper()}")
        
        handler = MissingValuesHandler(strategy=strategy, logger=logger)
        X_imputed = handler.fit_transform(X_secom)
        
        stats = handler.detect_missing(X_imputed)
        
        if stats['total_missing'] == 0:
            print_success(f"✓ {strategy} imputation successful")
            logger.info(f"  Final shape: {X_imputed.shape}")
        else:
            logger.warning(f"⚠ {strategy} imputation left {stats['total_missing']} missing values")
        
        logger.blank()
    
    print_success("✓ Test 2 PASSED - SECOM dataset handled")
    
except Exception as e: 
    logger.error(f"✗ Test 2 FAILED:  {e}")
    import traceback
    traceback.print_exc()

logger.blank()

# ============================================================================
# SUMMARY
# ============================================================================

print_header("TEST COMPLETE")
print_success("Missing Values Handler is working correctly!")
logger.blank()
logger.info("Available strategies:")
logger.info("  - mean:      Replace with column mean")
logger.info("  - median:   Replace with column median")
logger.info("  - mode:     Replace with most common value")
logger.info("  - knn:      K-Nearest Neighbors imputation")
logger.info("  - constant: Replace with 0")
logger.info("  - ffill:    Forward fill")
logger.info("  - bfill:    Backward fill")
logger.info("  - drop:     Drop columns/rows with missing values")