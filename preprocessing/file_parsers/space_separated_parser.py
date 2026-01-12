# ============================================================================
# SPACE-SEPARATED FILE PARSER
# ============================================================================
# Parse space-separated files (common in UCI datasets)
#
# Last updated: 2025-12-31
# ============================================================================

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from utils.logger import Logger


class SpaceSeparatedParser:
    """Parser for space-separated data files."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def parse(self,
             filepath:  Path,
             target_column: int = -1,
             has_header: bool = False,
             delimiter: str = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Parse space-separated or comma-separated file. 
        
        Args:
            filepath: Path to data file
            target_column: Index of target column (-1 = last)
            has_header: Whether file has header row
            delimiter: Delimiter (None = auto-detect, ' ' or ',')
            
        Returns: 
            (X, y, metadata) tuple
        """
        try:  
            if self.logger:
                self.logger.info(f"Parsing delimited file: {filepath. name}")
            
            # Auto-detect delimiter if not specified
            if delimiter is None:
                delimiter = self._detect_delimiter(filepath)
                if self.logger:
                    self. logger.debug(f"Detected delimiter: '{delimiter}'")
            
            # Read file
            if has_header:
                data = np.loadtxt(filepath, skiprows=1, delimiter=delimiter)
            else:
                data = np.loadtxt(filepath, delimiter=delimiter)
            
            if self.logger:
                self. logger.info(f"  Loaded:  {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Separate features and target
            if target_column == -1:
                X = data[: , :-1]
                y = data[:, -1]
            else:
                y = data[:, target_column]
                X = np.delete(data, target_column, axis=1)
            
            # Convert labels to integers if needed
            if not np.issubdtype(y. dtype, np.integer):
                # Check if labels are whole numbers
                if np.all(y == y.astype(int)):
                    y = y.astype(int)
            
            # Metadata
            metadata = {
                'filepath': str(filepath),
                'format': 'space_separated',
                'delimiter': delimiter,
                'samples': int(X.shape[0]),
                'features': int(X.shape[1]),
                'classes': int(len(np.unique(y))),
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            }
            
            if self.logger:
                self. logger.success(f"Parsed:  {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, metadata
            
        except Exception as e:
            if self.logger:
                self. logger.error(f"Delimited file parsing failed: {e}")
            raise
    
    def _detect_delimiter(self, filepath: Path) -> str:
        """
        Auto-detect delimiter (comma or space).
        
        Args:
            filepath: Path to file
            
        Returns:
            Delimiter character (',' or None for whitespace)
        """
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            # Count delimiters
            comma_count = first_line.count(',')
            space_count = first_line.count(' ')
            
            # If lots of commas, it's CSV-like
            if comma_count > 10:  # Arbitrary threshold
                return ','
            # If lots of spaces, it's space-separated
            elif space_count > 10:
                return None  # None = any whitespace
            # Default to comma if both are low but commas exist
            elif comma_count > 0: 
                return ','
            else:
                return None
                
        except Exception:
            # Default to whitespace
            return None
    
    def parse_secom(self,
                   features_file: Path,
                   labels_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:  
        """
        Parse SECOM dataset (separate features and labels files).
        
        Args:
            features_file: Path to secom. data
            labels_file: Path to secom_labels.data
            
        Returns:
            (X, y, metadata) tuple
        """
        try:  
            if self.logger:
                self. logger.info(f"Parsing SECOM dataset")
            
            # Load features
            X = np.loadtxt(features_file)
            
            # Load labels (ONLY the first column - ignore timestamp)
            # SECOM labels file format: [label, "timestamp"]
            try:
                # Method 1: Use usecols to only load first column
                y = np.loadtxt(labels_file, usecols=0)
            except Exception as e: 
                # Method 2: Fallback - robust parsing of mixed columns
                if self.logger:
                    self.logger.debug(f"Standard loading failed ({e}), using robust parser...")
                
                y = self._parse_mixed_columns(labels_file, target_column=0)
            
            # Convert labels:  -1 (Fail) → 0, 1 (Pass) → 1
            y = np.where(y == -1, 0, 1)
            
            if self.logger:
                self.logger.info(f"  Features:  {X.shape}")
                self.logger.info(f"  Labels: {y.shape}")
            
            # Check for missing values
            missing_count = np. isnan(X).sum()
            missing_percentage = (missing_count / X.size) * 100
            
            if self.logger and missing_count > 0:
                self. logger.warning(f"  Missing values: {missing_count} ({missing_percentage:.1f}%)")
            
            # Metadata
            metadata = {
                'filepath_features': str(features_file),
                'filepath_labels': str(labels_file),
                'format': 'secom',
                'samples': int(X.shape[0]),
                'features': int(X. shape[1]),
                'classes': 2,
                'class_names': ['Fail', 'Pass'],
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},  # ✅ Convert np.int32 → int
                'missing_values': int(missing_count),
                'missing_percentage': float(missing_percentage)
            }
            
            if self.logger:
                self.logger.success(f"Parsed SECOM:  {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, metadata
            
        except Exception as e:  
            if self.logger:
                self.logger.error(f"SECOM parsing failed: {e}")
            raise

    def _parse_mixed_columns(self, filepath: Path, target_column: int = 0) -> np.ndarray:
        """
        Parse file with mixed data types (numeric + strings).
        
        Useful for files like SECOM labels that have: 
          -1 "timestamp"
          1 "timestamp"
        
        Args:
            filepath: Path to file
            target_column:  Which column to extract (0-indexed)
            
        Returns: 
            Array of values from target column
        """
        values = []
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                
                # Split by whitespace
                parts = line.split()
                
                if len(parts) <= target_column:
                    if self.logger:
                        self.logger.warning(
                            f"Line {line_num}: Expected at least {target_column + 1} columns, "
                            f"got {len(parts)}"
                        )
                    continue
                
                try:
                    value = float(parts[target_column])
                    values.append(value)
                except ValueError: 
                    if self.logger:
                        self.logger.warning(
                            f"Line {line_num}: Could not convert '{parts[target_column]}' to float"
                        )
                    continue
        
        return np. array(values)