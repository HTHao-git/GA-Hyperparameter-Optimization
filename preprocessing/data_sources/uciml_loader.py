# ============================================================================
# UCI ML REPOSITORY LOADER (New API)
# ============================================================================
# Downloads datasets using the new ucimlrepo package
#
# Last updated: 2026-01-02
# ============================================================================

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from utils.logger import Logger


class UCIMLRepoLoader:
    """Loader for UCI ML Repository using new ucimlrepo package."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self._check_package()
    
    def _check_package(self) -> bool:
        """Check if ucimlrepo package is installed."""
        try:
            import ucimlrepo
            self.available = True
            return True
        except ImportError: 
            if self.logger:
                self.logger.debug("ucimlrepo package not installed")
            self.available = False
            return False
    
    def is_available(self) -> bool:
        """Check if loader is available."""
        return self.available
    
    def download_dataset(self,
                        dataset_id: int,
                        destination_dir: Path,
                        dataset_name: str = None) -> bool:
        """
        Download dataset from UCI ML Repository.
        
        Args:
            dataset_id: UCI dataset ID (e.g., 54 for ISOLET)
            destination_dir: Directory to save dataset
            dataset_name: Name for saved file (optional)
            
        Returns:
            True if successful
        """
        if not self.available:
            if self.logger:
                self.logger.error("ucimlrepo package not installed")
                self.logger.info("Install with:   pip install ucimlrepo")
            return False
        
        try:
            from ucimlrepo import fetch_ucirepo
            
            if self.logger:
                self.logger.info(f"Fetching dataset (ID={dataset_id}) from UCI ML Repo...")
            
            # Fetch dataset
            dataset = fetch_ucirepo(id=dataset_id)
            
            # Extract data
            X = dataset.data.features
            y = dataset.data.targets
            
            if self.logger:
                self.logger.success(f"Fetched:  {X.shape[0]} samples, {X.shape[1]} features")
            
            # Create destination directory
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV (easier to load later)
            if dataset_name is None:
                dataset_name = f"uci_dataset_{dataset_id}"
            
            # Combine X and y
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                combined = pd.concat([X, y], axis=1)
            else:
                # Convert to DataFrame if needed
                X_df = pd.DataFrame(X)
                y_df = pd.DataFrame(y, columns=['target'])
                combined = pd.concat([X_df, y_df], axis=1)
            
            # Save to CSV
            output_file = destination_dir / f"{dataset_name}.csv"
            combined.to_csv(output_file, index=False)
            
            if self.logger:
                self.logger.success(f"Saved to: {output_file}")
            
            # Also save metadata
            metadata_file = destination_dir / "uci_metadata.txt"
            with open(metadata_file, 'w') as f:
                f.write(f"Dataset ID: {dataset_id}\n")
                f.write(f"Name: {dataset.metadata.get('name', 'N/A')}\n")
                f.write(f"Abstract: {dataset.metadata.get('abstract', 'N/A')}\n")
                f.write(f"\nVariables:\n")
                if hasattr(dataset, 'variables'):
                    f.write(str(dataset.variables))
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"UCI ML Repo download failed: {e}")
            return False