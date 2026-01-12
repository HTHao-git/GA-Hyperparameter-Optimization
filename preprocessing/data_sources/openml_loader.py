# ============================================================================
# OPENML LOADER
# ============================================================================
# Downloads datasets from OpenML
#
# Last updated: 2026-01-02
# ============================================================================

from pathlib import Path
from typing import Optional
import numpy as np

from utils.logger import Logger


class OpenMLLoader:
    """Loader for OpenML datasets."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self._check_package()
    
    def _check_package(self) -> bool:
        """Check if openml package is installed."""
        try:
            import openml
            self.available = True
            return True
        except ImportError:
            if self.logger:
                self.logger.debug("openml package not installed")
            self.available = False
            return False
    
    def is_available(self) -> bool:
        """Check if loader is available."""
        return self.available
    
    def download_dataset(self,
                        dataset_id: int,
                        destination_dir: Path) -> bool:
        """
        Download dataset from OpenML.
        
        Args:
            dataset_id: OpenML dataset ID
            destination_dir: Directory to save dataset
            
        Returns:
            True if successful
        """
        if not self.available:
            if self.logger:
                self.logger.error("openml package not installed")
                self.logger.info("Install with: pip install openml")
            return False
        
        try: 
            import openml
            import pandas as pd
            
            if self.logger:
                self.logger.info(f"Downloading from OpenML (ID={dataset_id})...")
            
            # Download dataset
            dataset = openml.datasets.get_dataset(dataset_id)
            
            # Get data
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="array",
                target=dataset.default_target_attribute
            )
            
            if self.logger:
                self.logger.success(f"Downloaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Create destination directory
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays
            np.save(destination_dir / "X.npy", X)
            np.save(destination_dir / "y.npy", y)
            
            # Also save as CSV for easier inspection
            if attribute_names: 
                df = pd.DataFrame(X, columns=attribute_names)
            else:
                df = pd.DataFrame(X)
            
            df['target'] = y
            df.to_csv(destination_dir / "data.csv", index=False)
            
            if self.logger:
                self.logger.success(f"Saved to:  {destination_dir}")
            
            return True
            
        except Exception as e: 
            if self.logger:
                self.logger.error(f"OpenML download failed: {e}")
            return False