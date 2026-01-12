# ============================================================================
# TEST SCALING WITH REAL DATA
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler, MinMaxScaler, RobustScaler
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info
import numpy as np

logger = get_logger(name="SCALING_TEST", verbose=True)

print_header("SCALING TEST - SECOM DATASET")

# Load SECOM
loader = DatasetLoader(logger=logger, interactive=False)
X, y, metadata = loader.load_dataset('secom')

logger.blank()
logger.info(f"SECOM loaded:  {X.shape}")
logger.info(f"Missing values: {metadata.get('missing_percentage', 0):.1f}%")

# Handle missing values first
logger.blank()
print_section("Step 1: Handle Missing Values")

mv_handler = MissingValuesHandler(strategy='mean', logger=logger)
X_clean = mv_handler.fit_transform(X)

logger.blank()
print_info(f"After imputation: {X_clean.shape}")

# Split train/test (simple split for demo)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

logger.blank()
print_info(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Test each scaler
logger.blank()
print_section("Step 2: Test Scalers")

scalers = [
    ('StandardScaler', StandardScaler(logger=logger)),
    ('MinMaxScaler', MinMaxScaler(logger=logger)),
    ('RobustScaler', RobustScaler(logger=logger))
]

for name, scaler in scalers:
    logger.blank()
    logger.section(name)
    
    # Fit on train, transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check statistics
    logger.info(f"Train scaled: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    logger.info(f"Test scaled:   mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")
    
    print_success(f"✓ {name} applied successfully")

logger.blank()
print_header("TEST COMPLETE")
print_success("Scaling module works with real data!")