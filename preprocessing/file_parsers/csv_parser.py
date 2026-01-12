# ============================================================================
# CSV FILE PARSER
# ============================================================================
# Parse CSV files and similar formats
#
# Last updated: 2025-12-31
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from utils.logger import Logger


class CSVParser:
    """Parser for CSV and similar delimited files."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def parse(self,
             filepath: Path,
             delimiter: str = ',',
             has_header: bool = True,
             target_column: Optional[str] = None,
             exclude_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Parse CSV file into X, y arrays.
        
        Args:
            filepath: Path to CSV file
            delimiter: Column delimiter (default: ',')
            has_header: Whether file has header row
            target_column: Name or index of target column
            exclude_columns: List of columns to exclude
            
        Returns: 
            (X, y, metadata) tuple
        """
        try: 
            if self.logger:
                self.logger.info(f"Parsing CSV:  {filepath.name}")
            
            # Read CSV
            if has_header:
                df = pd.read_csv(filepath, delimiter=delimiter)
            else:
                df = pd.read_csv(filepath, delimiter=delimiter, header=None)
            
            if self.logger:
                self.logger.info(f"  Loaded:  {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Exclude columns
            if exclude_columns:
                df = df.drop(columns=exclude_columns, errors='ignore')
            
            # Separate features and target
            if target_column is not None:
                # Target specified
                if isinstance(target_column, str):
                    y = df[target_column].values
                    X = df.drop(columns=[target_column]).values
                else:  # Integer index
                    y = df.iloc[:, target_column].values
                    X = df.drop(df.columns[target_column], axis=1).values
            else:
                # Assume last column is target
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1].values
            
            # Metadata
            metadata = {
                'filepath': str(filepath),
                'format': 'csv',
                'delimiter': delimiter,
                'has_header': has_header,
                'samples':  int(X.shape[0]),
                'features': int(X.shape[1]),
                'classes': int(len(np. unique(y))),
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            }
            
            if self.logger:
                self.logger.success(f"Parsed:  {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"CSV parsing failed: {e}")
            raise


    def auto_detect_target(self, filepath: Path) -> Optional[str]:
        """
        Auto-detect target column (heuristics).
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Target column name or None
        """
        try:
            df = pd.read_csv(filepath, nrows=100)  # Sample first 100 rows
            
            # Common target column names
            target_names = ['label', 'target', 'class', 'y', 'output', 'category']
            
            for col in df.columns:
                if col.lower() in target_names:
                    return col
            
            # If not found, assume last column
            return df.columns[-1]
            
        except: 
            return None