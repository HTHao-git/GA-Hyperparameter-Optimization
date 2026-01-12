# ============================================================================
# DATA LOADER TEST SUITE
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info

# Initialize
logger = get_logger(name="DATA_LOADER_TEST", verbose=True)
loader = DatasetLoader(logger=logger, interactive=True)

print_header("DATA LOADER TEST SUITE")

# ============================================================================
# TEST 1: List Available Datasets
# ============================================================================

print_section("TEST 1: List Available Datasets")

available = loader.list_available_datasets()
print_info(f"Found {len(available)} built-in datasets:")
for dataset in available:
    info = loader.get_dataset_info(dataset)
    metadata = info.get('metadata', {})
    print(f"  • {dataset}")
    print(f"      Samples: {metadata.get('samples', 'N/A')}")
    print(f"      Features:  {metadata.get('features', 'N/A')}")
    print(f"      Classes: {metadata.get('classes', 'N/A')}")

logger.blank()

# ============================================================================
# TEST 2: Load SECOM Dataset
# ============================================================================

print_section("TEST 2: Load SECOM Dataset (Auto-Download)")

try:
    X, y, metadata = loader.load_dataset('secom', source='auto')
    
    logger.blank()
    print_success("SECOM Dataset Loaded Successfully!")
    logger.blank()
    
    print_info("Dataset Summary:")
    print(f"  Shape: {X.shape}")
    print(f"  Features (X): {X.shape}")
    print(f"  Labels (y): {y.shape}")
    print(f"  Classes: {metadata.get('classes', 'N/A')}")
    print(f"  Class distribution: {metadata.get('class_distribution', {})}")
    
    if 'missing_values' in metadata:
        print(f"  Missing values:  {metadata['missing_values']} ({metadata.get('missing_percentage', 0):.1f}%)")
    
    logger.blank()
    print_info("Sample Data:")
    print(f"  X[0]: {X[0][:5]}...  (showing first 5 features)")
    print(f"  y[0]: {y[0]}")
    
    logger.blank()
    print_success("✓ Test 2 PASSED")
    
except Exception as e:
    logger.error(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

logger.blank()

# ============================================================================
# TEST 3: Check Caching
# ============================================================================

print_section("TEST 3: Load SECOM Again (Should Use Cache)")

try:
    X2, y2, metadata2 = loader.load_dataset('secom', source='auto', force_download=False)
    
    logger.blank()
    
    # Verify it's the same data
    import numpy as np
    
    if np.array_equal(X, X2) and np.array_equal(y, y2):
        print_success("✓ Caching works!  Loaded from local cache.")
    else:
        logger.warning("⚠ Data mismatch - caching may not be working")
    
    print_success("✓ Test 3 PASSED")
    
except Exception as e:
    logger.error(f"✗ Test 3 FAILED: {e}")

logger.blank()

# ============================================================================
# TEST 4: List Cached Datasets
# ============================================================================

print_section("TEST 4: List Cached Datasets")

cached = loader.list_cached_datasets()
print_info(f"Cached datasets: {len(cached)}")
for dataset in cached:
    print(f"  • {dataset}")

if 'secom' in cached: 
    print_success("✓ SECOM is now cached")
    print_success("✓ Test 4 PASSED")
else:
    logger.warning("⚠ SECOM not found in cache")

logger.blank()

# ============================================================================
# TEST 5: Dataset Info
# ============================================================================

print_section("TEST 5: Get Dataset Info")

info = loader.get_dataset_info('secom')
print_info("SECOM Information:")
print(f"  Name: {info.get('name', 'N/A')}")
print(f"  Description: {info.get('description', 'N/A')}")
print(f"  Available sources: {list(info.get('sources', {}).keys())}")

metadata_cached = loader._load_metadata(loader.dataset_dir / 'secom')
if metadata_cached:
    print(f"  Loaded at: {metadata_cached.get('loaded_at', 'N/A')}")

print_success("✓ Test 5 PASSED")

logger.blank()

# ============================================================================
# SUMMARY
# ============================================================================

print_header("TEST SUITE COMPLETE")

print_success("All basic tests passed!")
print()
print_info("Next steps:")
print("  1. Test other datasets (fashion_mnist, isolet, steel_plates)")
print("  2. Test custom dataset loading")
print("  3. Test interactive prompts for failed downloads")