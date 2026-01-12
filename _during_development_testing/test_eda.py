# ============================================================================
# TEST EDA WITH REAL DATA (SECOM)
# ============================================================================

from preprocessing.data_loader import DatasetLoader
from preprocessing.eda import EDA
from utils.logger import get_logger
from utils.colors import print_header
from pathlib import Path

logger = get_logger(name="EDA_TEST", verbose=True)

print_header("EDA TEST - SECOM DATASET")

# Load SECOM
loader = DatasetLoader(logger=logger, interactive=False)
X, y, metadata = loader.load_dataset('secom')

logger.blank()

# Create EDA instance
eda = EDA(X, y, logger=logger)

# Generate comprehensive report
output_dir = Path('outputs/eda_secom')
eda.generate_report(output_dir=output_dir)

logger.blank()
print_header("EDA COMPLETE")
logger.info(f"Visualizations saved to:  {output_dir}")