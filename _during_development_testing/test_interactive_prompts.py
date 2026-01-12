# ============================================================================
# TEST INTERACTIVE PROMPTS
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger

logger = get_logger(name="INTERACTIVE_TEST", verbose=True)

# Create loader
loader = DatasetLoader(logger=logger, interactive=True)

# Try loading a dataset that doesn't exist locally
# This will trigger interactive prompts

print("="*70)
print("INTERACTIVE PROMPT TEST")
print("="*70)
print()
print("This test will:")
print("  1. Try to load 'steel_plates' dataset")
print("  2. If not cached, attempt download")
print("  3. If download fails, show interactive prompts")
print()
print("You can test:")
print("  - Browsing for files (option 2)")
print("  - Entering file paths (option 3)")
print("  - Skipping dataset (option 4)")
print()
input("Press Enter to continue...")

try:
    X, y, metadata = loader.load_dataset('steel_plates', source='auto')
    
    print()
    print("="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Dataset loaded:  {X.shape[0]} samples, {X.shape[1]} features")
    
except Exception as e:
    print()
    print("="*70)
    print("RESULT")
    print("="*70)
    print(f"Dataset loading ended:  {e}")
    print("This is expected if you chose to skip or no file was provided.")