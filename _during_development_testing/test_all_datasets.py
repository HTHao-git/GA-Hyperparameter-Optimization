# ============================================================================
# TEST ALL DATASETS - Download and Cache
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_error, print_info
import numpy as np

# Initialize
logger = get_logger(name="DATASET_TEST", verbose=True)
loader = DatasetLoader(logger=logger, interactive=True)

print_header("TESTING ALL DATASETS")

# Track results
results = {
    'secom': {'status': 'pending', 'error': None},
    'fashion_mnist': {'status': 'pending', 'error': None},
    'isolet': {'status':  'pending', 'error':  None},
    'steel_plates': {'status': 'pending', 'error': None}
}

# ============================================================================
# TEST 1: SECOM (Already tested, but let's verify)
# ============================================================================

print_section("TEST 1: SECOM Dataset")

try:
    X, y, metadata = loader.load_dataset('secom', source='auto')
    
    logger.blank()
    print_success("✓ SECOM loaded successfully")
    print_info(f"  Shape: {X.shape}")
    print_info(f"  Classes: {metadata['classes']}")
    print_info(f"  Missing values: {metadata.get('missing_values', 0)} ({metadata.get('missing_percentage', 0):.1f}%)")
    print_info(f"  Class distribution: {metadata['class_distribution']}")
    
    results['secom']['status'] = 'success'
    results['secom']['shape'] = X.shape
    results['secom']['metadata'] = metadata
    
except Exception as e:
    logger.error(f"✗ SECOM failed: {e}")
    results['secom']['status'] = 'failed'
    results['secom']['error'] = str(e)

logger.blank()

# ============================================================================
# TEST 2: Fashion-MNIST (TensorFlow download)
# ============================================================================

print_section("TEST 2: Fashion-MNIST Dataset")

try:
    X, y, metadata = loader.load_dataset('fashion_mnist', source='auto')
    
    logger.blank()
    print_success("✓ Fashion-MNIST loaded successfully")
    print_info(f"  Shape: {X.shape}")
    print_info(f"  Classes: {metadata['classes']}")
    print_info(f"  Train samples: {metadata.get('train_samples', 'N/A')}")
    print_info(f"  Test samples: {metadata.get('test_samples', 'N/A')}")
    print_info(f"  Class distribution: {metadata['class_distribution']}")
    
    results['fashion_mnist']['status'] = 'success'
    results['fashion_mnist']['shape'] = X.shape
    results['fashion_mnist']['metadata'] = metadata
    
except Exception as e:
    logger.error(f"✗ Fashion-MNIST failed: {e}")
    results['fashion_mnist']['status'] = 'failed'
    results['fashion_mnist']['error'] = str(e)
    import traceback
    traceback.print_exc()

logger.blank()

# ============================================================================
# TEST 3: ISOLET (UCI download)
# ============================================================================

print_section("TEST 3: ISOLET Dataset")

try:
    X, y, metadata = loader.load_dataset('isolet', source='auto')
    
    logger.blank()
    print_success("✓ ISOLET loaded successfully")
    print_info(f"  Shape: {X.shape}")
    print_info(f"  Classes: {metadata['classes']} (A-Z letters)")
    print_info(f"  Train samples: {metadata.get('train_samples', 'N/A')}")
    print_info(f"  Test samples: {metadata.get('test_samples', 'N/A')}")
    print_info(f"  Class distribution (first 5): {dict(list(metadata['class_distribution'].items())[:5])}...")
    
    results['isolet']['status'] = 'success'
    results['isolet']['shape'] = X.shape
    results['isolet']['metadata'] = metadata
    
except Exception as e:
    logger.error(f"✗ ISOLET failed: {e}")
    results['isolet']['status'] = 'failed'
    results['isolet']['error'] = str(e)
    import traceback
    traceback.print_exc()

logger.blank()

# ============================================================================
# TEST 4: Steel Plates Faults (Kaggle or UCI)
# ============================================================================

print_section("TEST 4: Steel Plates Faults Dataset")

try:
    X, y, metadata = loader.load_dataset('steel_plates', source='auto')
    
    logger.blank()
    print_success("✓ Steel Plates loaded successfully")
    print_info(f"  Shape: {X.shape}")
    print_info(f"  Classes: {metadata['classes']}")
    print_info(f"  Class names: {metadata.get('class_names', 'N/A')}")
    print_info(f"  Class distribution: {metadata['class_distribution']}")
    
    results['steel_plates']['status'] = 'success'
    results['steel_plates']['shape'] = X.shape
    results['steel_plates']['metadata'] = metadata
    
except Exception as e:
    logger.error(f"✗ Steel Plates failed: {e}")
    results['steel_plates']['status'] = 'failed'
    results['steel_plates']['error'] = str(e)
    import traceback
    traceback.print_exc()

logger.blank()

# ============================================================================
# SUMMARY
# ============================================================================

print_header("DATASET TEST SUMMARY")

print_info("Results:")
print()

success_count = 0
failed_count = 0

for dataset, result in results.items():
    status = result['status']
    
    if status == 'success':
        print_success(f"✓ {dataset:15} - SUCCESS - Shape: {result['shape']}")
        success_count += 1
    elif status == 'failed':
        print_error(f"✗ {dataset:15} - FAILED - {result['error']}")
        failed_count += 1
    else:
        logger.warning(f"⚠ {dataset:15} - PENDING")

logger.blank()

print_info(f"Success: {success_count}/4")
if failed_count > 0:
    print_error(f"Failed:   {failed_count}/4")
else:
    print_success("All datasets loaded successfully!")

logger.blank()

# Show cached datasets
print_section("Cached Datasets")

cached = loader.list_cached_datasets()
print_info(f"Total cached:  {len(cached)}")
for dataset in cached:
    print(f"  • {dataset}")

logger.blank()

if success_count == 4:
    print_header("🎉 ALL DATASETS READY!")
    print()
    print_success("All datasets downloaded and cached successfully!")
    print_info("You can now proceed to Phase 2.2 with confidence.")
else:
    print_header("⚠️ SOME DATASETS FAILED")
    print()
    logger.warning(f"{failed_count} dataset(s) failed to load.")
    logger.info("You can:")
    logger.info("  1. Try manual download for failed datasets")
    logger.info("  2. Proceed with available datasets")
    logger.info("  3. Debug the errors above")