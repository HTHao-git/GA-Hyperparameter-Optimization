# Test loading custom dataset

from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger
from utils.colors import print_header, print_success, print_info
import numpy as np

logger = get_logger(name="CUSTOM_TEST", verbose=True)
loader = DatasetLoader(logger=logger, interactive=True)

print_header("CUSTOM DATASET LOADING TEST")

try:
    # Load the custom dataset we just created
    X, y, metadata = loader.load_dataset('custom/my_test_dataset', source='local')
    
    print()
    print_success("Custom dataset loaded!")
    print()
    print_info("Dataset Summary:")
    print(f"  Shape: {X.shape}")
    print(f"  Features:  {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    print()
    print_success("✓ Custom dataset test PASSED")
    
except Exception as e:
    logger.error(f"✗ Test FAILED: {e}")
    import traceback
    traceback.print_exc()