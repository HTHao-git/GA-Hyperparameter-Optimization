# ============================================================================
# AUTO FORMAT DETECTOR & PARSER
# ============================================================================
# Automatically detect file format and use appropriate parser
#
# Last updated: 2025-12-31
# ============================================================================

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

from utils.logger import Logger
from preprocessing.file_parsers.csv_parser import CSVParser
from preprocessing.file_parsers.space_separated_parser import SpaceSeparatedParser
from preprocessing.file_parsers.nna_parser import NNAParser


class AutoParser:
    """Automatically detect format and parse file."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.csv_parser = CSVParser(logger)
        self.space_parser = SpaceSeparatedParser(logger)
        self.nna_parser = NNAParser(logger)
    
    def detect_format(self, filepath: Path) -> str:
        """
        Detect file format based on extension and content.
        
        Args:
            filepath: Path to file
            
        Returns:
            Format string ('csv', 'space_separated', 'nna', 'npz', 'unknown')
        """
        extension = filepath.suffix.lower()
        
        # Extension-based detection
        format_map = {
            '.csv': 'csv',
            '.txt': 'auto',  # Could be CSV or space-separated
            '.data': 'space_separated',
            '.nna': 'nna',
            '.npz': 'npz',
            '.arff': 'arff',
            '.xlsx': 'excel',
            '.xls': 'excel',
        }
        
        detected = format_map.get(extension, 'unknown')
        
        # If .txt, try to detect delimiter
        if detected == 'auto':
            detected = self._detect_delimiter(filepath)
        
        if self.logger:
            self.logger.info(f"Detected format: {detected}")
        
        return detected
    
    def _detect_delimiter(self, filepath: Path) -> str:
        """
        Detect delimiter by sampling first few lines.
        
        Args:
            filepath: Path to file
            
        Returns:
            'csv' or 'space_separated'
        """
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline()
            
            # Count delimiters
            comma_count = first_line.count(',')
            space_count = first_line.count(' ')
            tab_count = first_line.count('\t')
            
            if comma_count > space_count and comma_count > tab_count:
                return 'csv'
            elif tab_count > comma_count and tab_count > space_count:
                return 'csv_tab'
            else:
                return 'space_separated'
        except: 
            return 'unknown'
    
    def parse(self,
             filepath: Path,
             format_hint: Optional[str] = None,
             **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """
        Auto-detect format and parse file.
        
        Args:
            filepath: Path to file
            format_hint: Optional format hint (overrides auto-detection)
            **kwargs: Additional arguments passed to parser
            
        Returns:
            (X, y, metadata) tuple
        """
        # Detect format
        if format_hint:
            file_format = format_hint
        else:
            file_format = self.detect_format(filepath)
        
        # Parse based on format
        if file_format == 'csv':
            return self.csv_parser.parse(filepath, delimiter=',', **kwargs)
        
        elif file_format == 'csv_tab':
            return self.csv_parser.parse(filepath, delimiter='\t', **kwargs)
        
        elif file_format == 'space_separated': 
            return self.space_parser.parse(filepath, **kwargs)
        
        elif file_format == 'nna':
            return self.nna_parser.parse(filepath)
        
        elif file_format == 'npz':
            return self._parse_npz(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def _parse_npz(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Parse .npz file (NumPy compressed format)."""
        try:
            data = np.load(filepath)
            
            # Fashion-MNIST format
            if 'X_train' in data and 'X_test' in data:
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
                
                # Flatten images if needed
                if X_train.ndim == 3:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Combine train and test
                X = np.vstack([X_train, X_test])
                y = np.hstack([y_train, y_test])
                
                metadata = {
                    'filepath':  str(filepath),
                    'format': 'npz',
                    'samples': int(X.shape[0]),
                    'train_samples': int(X_train.shape[0]),
                    'test_samples': int(X_test.shape[0]),
                    'features': int(X.shape[1]),
                    'classes': int(len(np.unique(y))),
                    'class_distribution':  {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
                }
                
                if self.logger:
                    self.logger.success(f"Parsed NPZ: {X.shape[0]} samples, {X.shape[1]} features")
                
                return X, y, metadata
            
            else:
                raise ValueError("NPZ file format not recognized")
        
        except Exception as e: 
            if self.logger:
                self.logger.error(f"NPZ parsing failed: {e}")
            raise