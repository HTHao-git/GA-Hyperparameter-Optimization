# ============================================================================
# TEST PCA WITH REAL DATA (SECOM)
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler
from preprocessing.pca import PCA
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info
import numpy as np

logger = get_logger(name="PCA_TEST", verbose=True)

print_header("PCA TEST - SECOM DATASET")

# ============================================================================
# LOAD AND PREPROCESS
# ============================================================================

loader = DatasetLoader(logger=logger, interactive=False)
X, y, metadata = loader.load_dataset('secom')

logger.blank()
print_info(f"Original SECOM:  {X.shape}")

# Preprocessing pipeline
logger.blank()
print_section("Preprocessing Pipeline")

# Step 1: Handle missing values
logger.info("Step 1: Imputing missing values...")
mv_handler = MissingValuesHandler(strategy='mean', logger=logger)
X_clean = mv_handler.fit_transform(X)

logger.blank()

# Step 2: Scale features (required for PCA!)
logger.info("Step 2: Scaling features...")
scaler = StandardScaler(logger=logger)
X_scaled = scaler.fit_transform(X_clean)

logger.blank()

# ============================================================================
# TEST DIFFERENT PCA STRATEGIES
# ============================================================================

print_section("PCA Dimensionality Reduction")

# Test 1: 95% variance
logger.section("Strategy 1: Keep 95% variance")
pca_95 = PCA(n_components=0.95, logger=logger)
X_pca_95 = pca_95.fit_transform(X_scaled)
logger.blank()
pca_95.report_variance()
logger.blank()

# Test 2: 99% variance
logger.section("Strategy 2: Keep 99% variance")
pca_99 = PCA(n_components=0.99, logger=logger)
X_pca_99 = pca_99.fit_transform(X_scaled)
logger.blank()

# Test 3: Fixed 50 components
logger.section("Strategy 3: Keep exactly 50 components")
pca_50 = PCA(n_components=50, logger=logger)
X_pca_50 = pca_50.fit_transform(X_scaled)
logger.blank()

# Test 4: Fixed 20 components (aggressive)
logger.section("Strategy 4: Keep exactly 20 components (aggressive)")
pca_20 = PCA(n_components=20, logger=logger)
X_pca_20 = pca_20.fit_transform(X_scaled)
logger.blank()

# ============================================================================
# COMPARISON
# ============================================================================

print_header("DIMENSIONALITY REDUCTION COMPARISON")

results = [
    ("Original", X_scaled.shape, None),
    ("95% variance", X_pca_95.shape, pca_95.explained_variance_ratio_.sum() * 100),
    ("99% variance", X_pca_99.shape, pca_99.explained_variance_ratio_.sum() * 100),
    ("50 components", X_pca_50.shape, pca_50.explained_variance_ratio_.sum() * 100),
    ("20 components", X_pca_20.shape, pca_20.explained_variance_ratio_.sum() * 100),
]

print_info("Results:")
print()

for name, shape, variance in results:
    if variance is None:
        print(f"  {name: <20} - Shape: {shape}")
    else:
        reduction = (1 - shape[1] / 590) * 100
        print(f"  {name:<20} - Shape: {shape} - Variance: {variance: 5.2f}% - Reduction: {reduction:5.1f}%")

logger.blank()

# ============================================================================
# RECONSTRUCTION TEST
# ============================================================================

print_section("Reconstruction Quality Test")

X_reconstructed = pca_95.inverse_transform(X_pca_95)
X_reconstructed = scaler.inverse_transform(X_reconstructed)

# Compare to original (with imputed values)
mse = np.mean((X_clean - X_reconstructed) ** 2)
relative_error = mse / np.var(X_clean) * 100

logger.info(f"Reconstruction MSE: {mse:.6f}")
logger.info(f"Relative error: {relative_error:.2f}%")

if relative_error < 10:
    print_success("✓ Good reconstruction quality")
else:
    logger.warning(f"⚠ Moderate reconstruction error: {relative_error:.2f}%")

logger.blank()

# ============================================================================
# COMPONENT IMPORTANCE
# ============================================================================

print_section("Top Component Analysis")

# Show top 3 components
pca_95.report_component_importance(n_components=3, n_features=5)

logger.blank()

# ============================================================================
# SUMMARY
# ============================================================================

print_header("SUMMARY")

print_success("✓ PCA successfully applied to SECOM!")
print()
print_info("Recommendations:")
print(f"  • 95% variance:  {pca_95.n_components_} components (good balance)")
print(f"  • 99% variance: {pca_99.n_components_} components (preserve more info)")
print(f"  • Fixed 50: preserves {pca_50.explained_variance_ratio_.sum() * 100:.2f}% variance")
print(f"  • Fixed 20: preserves {pca_20.explained_variance_ratio_.sum() * 100:.2f}% variance (aggressive)")
print()
print_info(f"Recommended:  Use {pca_95.n_components_} components (95% variance) for good balance")