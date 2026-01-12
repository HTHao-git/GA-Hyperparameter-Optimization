# ============================================================================
# NNA FILE PARSER
# ============================================================================
# Parse .NNA files (Steel Plates Faults dataset format)
#
# Last updated: 2025-12-31
# ============================================================================

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from utils.logger import Logger


class NNAParser:
    """Parser for .NNA files (Steel Plates custom format)."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def parse(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Parse .NNA file.
        
        The NNA format is a custom format where: 
        - First 27 columns are features
        - Last 7 columns are binary labels for 7 fault types
        - Each row can have multiple labels (multi-label)
        - For multi-class, we take the first active label
        
        Args: 
            filepath: Path to .NNA file
            
        Returns: 
            (X, y, metadata) tuple
        """
        try: 
            if self.logger:
                self.logger.info(f"Parsing NNA file: {filepath.name}")
            
            # Read file (tab-separated or space-separated)
            # Try tab first
            try:
                data = np.loadtxt(filepath, delimiter='\t')
            except: 
                # Fall back to space-separated
                data = np.loadtxt(filepath)
            
            if self.logger:
                self.logger.info(f"  Loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Steel Plates:  27 features + 7 binary labels
            if data.shape[1] == 34: 
                X = data[:, :27]  # Features
                labels_multi = data[:, 27:]  # 7 binary labels
                
                # Convert multi-label to multi-class
                # Take the first active label (1) as the class
                y = np.argmax(labels_multi, axis=1)
                
                # Handle rows with no active labels (shouldn't happen, but just in case)
                no_label_mask = labels_multi.sum(axis=1) == 0
                if no_label_mask.any():
                    if self.logger:
                        self.logger.warning(f"  {no_label_mask.sum()} samples have no labels")
                    y[no_label_mask] = -1  # Mark as unknown
            
            else:
                # Unknown format, assume last column is target
                if self.logger:
                    self.logger.warning(f"Unexpected NNA format ({data.shape[1]} columns)")
                    self.logger.info("Assuming last column is target")
                
                X = data[:, :-1]
                y = data[:, -1]. astype(int)
            
            # Metadata
            class_names = [
                'Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 
                'Dirtiness', 'Bumps', 'Other_Faults'
            ]
            
            metadata = {
                'filepath': str(filepath),
                'format': 'nna',
                'samples':  int(X.shape[0]),
                'features': int(X.shape[1]),
                'classes': int(len(np. unique(y[y >= 0]))),
                'class_names': class_names,
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            }
            
            if self.logger:
                self.logger.success(f"Parsed NNA: {X.shape[0]} samples, {X.shape[1]} features, {metadata['classes']} classes")
            
            return X, y, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"NNA parsing failed: {e}")
            raise