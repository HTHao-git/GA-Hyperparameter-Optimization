# ============================================================================
# VALIDATORS - Input Validation Helpers
# ============================================================================
# This module provides validation functions for user inputs, file formats,
# data ranges, and configuration values. 
#
# USAGE:
#   from utils. validators import validate_file_path, validate_range
#   
#   validate_file_path("data.csv", must_exist=True)
#   validate_range(0. 5, 0.0, 1.0, "dropout")
#
# FEATURES:
#   - File path validation (existence, extension, readability)
#   - Numeric range validation
#   - Type validation
#   - Choice validation (categorical values)
#   - Dataset format detection
#
# Last updated: 2025-12-31
# ============================================================================

import os
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from utils.colors import Colors, print_error, print_warning


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_file_path(filepath: Union[str, Path],
                      must_exist: bool = True,
                      allowed_extensions:  Optional[List[str]] = None,
                      check_readable: bool = True) -> Path:
    """
    Validate file path. 
    
    Args:
        filepath: Path to file
        must_exist: If True, raise error if file doesn't exist
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.data'])
        check_readable: If True, check if file is readable
        
    Returns:
        Path object (validated)
        
    Raises: 
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If file extension is not allowed
        PermissionError: If file is not readable
        
    Example:
        >>> validate_file_path("data.csv", allowed_extensions=['.csv'])
        PosixPath('data.csv')
    """
    filepath = Path(filepath)
    
    # Check existence
    if must_exist and not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check extension
    if allowed_extensions is not None:
        if filepath.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(
                f"Invalid file extension '{filepath.suffix}'. "
                f"Allowed:  {allowed_extensions}"
            )
    
    # Check readable
    if check_readable and filepath.exists():
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"File is not readable: {filepath}")
    
    return filepath


def validate_directory(dirpath: Union[str, Path],
                      create_if_missing: bool = False) -> Path:
    """
    Validate directory path.
    
    Args:
        dirpath:  Path to directory
        create_if_missing: If True, create directory if it doesn't exist
        
    Returns:
        Path object (validated)
        
    Raises: 
        NotADirectoryError: If path exists but is not a directory
        FileNotFoundError: If directory doesn't exist and create_if_missing=False
        
    Example:
        >>> validate_directory("outputs/", create_if_missing=True)
        PosixPath('outputs')
    """
    dirpath = Path(dirpath)
    
    if dirpath.exists():
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {dirpath}")
    else:
        if create_if_missing:
            dirpath. mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {dirpath}")
    
    return dirpath


def detect_file_format(filepath: Union[str, Path]) -> str:
    """
    Detect file format based on extension and content.
    
    Args:
        filepath: Path to file
        
    Returns:
        File format string ('csv', 'data', 'arff', 'excel', 'unknown')
        
    Example: 
        >>> detect_file_format("data.csv")
        'csv'
    """
    filepath = Path(filepath)
    extension = filepath.suffix.lower()
    
    # Map extensions to formats
    format_map = {
        '.csv': 'csv',
        '.txt': 'csv',  # Assume CSV if . txt
        '.data': 'data',
        '.arff': 'arff',
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.json': 'json',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
    }
    
    return format_map.get(extension, 'unknown')


# ============================================================================
# NUMERIC VALIDATION
# ============================================================================

def validate_range(value: Union[int, float],
                  min_val: Optional[Union[int, float]] = None,
                  max_val: Optional[Union[int, float]] = None,
                  param_name: str = "value",
                  inclusive: bool = True) -> bool:
    """
    Validate that a numeric value is within a range.
    
    Args:
        value: Value to validate
        min_val:  Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        param_name:  Parameter name (for error messages)
        inclusive: If True, use <= and >=; if False, use < and >
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If value is out of range
        
    Example: 
        >>> validate_range(0.5, 0.0, 1.0, "dropout")
        True
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be numeric, got {type(value)}")
    
    if inclusive:
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name}={value} is below minimum {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name}={value} exceeds maximum {max_val}")
    else:
        if min_val is not None and value <= min_val:
            raise ValueError(f"{param_name}={value} must be greater than {min_val}")
        if max_val is not None and value >= max_val:
            raise ValueError(f"{param_name}={value} must be less than {max_val}")
    
    return True


def validate_positive(value: Union[int, float],
                     param_name: str = "value",
                     allow_zero: bool = False) -> bool:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        param_name: Parameter name (for error messages)
        allow_zero: If True, allow zero
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If value is not positive
        
    Example: 
        >>> validate_positive(5, "population_size")
        True
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be numeric, got {type(value)}")
    
    if allow_zero: 
        if value < 0:
            raise ValueError(f"{param_name}={value} must be non-negative")
    else:
        if value <= 0:
            raise ValueError(f"{param_name}={value} must be positive")
    
    return True


def validate_integer(value: Any,
                    param_name:  str = "value") -> bool:
    """
    Validate that a value is an integer.
    
    Args:
        value: Value to validate
        param_name: Parameter name (for error messages)
        
    Returns:
        True if valid
        
    Raises:
        TypeError: If value is not an integer
        
    Example:
        >>> validate_integer(5, "num_layers")
        True
    """
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{param_name} must be an integer, got {type(value)}")
    
    return True


# ============================================================================
# CHOICE VALIDATION
# ============================================================================

def validate_choice(value: Any,
                   choices: List[Any],
                   param_name:  str = "value",
                   case_sensitive: bool = True) -> bool:
    """
    Validate that a value is in a list of allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed values
        param_name: Parameter name (for error messages)
        case_sensitive: If False, ignore case for string comparison
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If value is not in choices
        
    Example:
        >>> validate_choice("relu", ["relu", "tanh", "elu"], "activation")
        True
    """
    if not case_sensitive and isinstance(value, str):
        # Case-insensitive comparison for strings
        choices_lower = [str(c).lower() for c in choices]
        if value.lower() not in choices_lower: 
            raise ValueError(
                f"{param_name}='{value}' is not a valid choice. "
                f"Allowed:  {choices}"
            )
    else:
        if value not in choices: 
            raise ValueError(
                f"{param_name}='{value}' is not a valid choice. "
                f"Allowed: {choices}"
            )
    
    return True


# ============================================================================
# TYPE VALIDATION
# ============================================================================

def validate_type(value: Any,
                 expected_type:  Union[type, Tuple[type, ...]],
                 param_name: str = "value") -> bool:
    """
    Validate that a value is of expected type. 
    
    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        param_name: Parameter name (for error messages)
        
    Returns:
        True if valid
        
    Raises:
        TypeError: If value is not of expected type
        
    Example:
        >>> validate_type("SECOM", str, "dataset_name")
        True
        >>> validate_type(0.5, (int, float), "learning_rate")
        True
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join([t.__name__ for t in expected_type])
            raise TypeError(
                f"{param_name} must be {type_names}, got {type(value).__name__}"
            )
        else:
            raise TypeError(
                f"{param_name} must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    return True


# ============================================================================
# ARRAY/LIST VALIDATION
# ============================================================================

def validate_list_length(lst: List[Any],
                        expected_length: Optional[int] = None,
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        param_name: str = "list") -> bool:
    """
    Validate list length.
    
    Args:
        lst: List to validate
        expected_length: Expected exact length (None = no requirement)
        min_length:  Minimum length (None = no minimum)
        max_length: Maximum length (None = no maximum)
        param_name: Parameter name (for error messages)
        
    Returns: 
        True if valid
        
    Raises:
        ValueError: If list length is invalid
        
    Example:
        >>> validate_list_length([1, 2, 3], min_length=2, max_length=5)
        True
    """
    if not isinstance(lst, (list, tuple, np.ndarray)):
        raise TypeError(f"{param_name} must be a list/array, got {type(lst)}")
    
    length = len(lst)
    
    if expected_length is not None:
        if length != expected_length:
            raise ValueError(
                f"{param_name} must have length {expected_length}, got {length}"
            )
    
    if min_length is not None and length < min_length: 
        raise ValueError(
            f"{param_name} must have at least {min_length} elements, got {length}"
        )
    
    if max_length is not None and length > max_length: 
        raise ValueError(
            f"{param_name} must have at most {max_length} elements, got {length}"
        )
    
    return True


def validate_array_shape(array: np.ndarray,
                        expected_shape: Optional[Tuple[int, ...]] = None,
                        expected_ndim: Optional[int] = None,
                        param_name:  str = "array") -> bool:
    """
    Validate numpy array shape.
    
    Args:
        array: Array to validate
        expected_shape: Expected shape (None = no requirement, -1 = any size)
        expected_ndim: Expected number of dimensions (None = no requirement)
        param_name: Parameter name (for error messages)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If array shape is invalid
        
    Example:
        >>> validate_array_shape(np. array([[1, 2], [3, 4]]), expected_ndim=2)
        True
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{param_name} must be a numpy array, got {type(array)}")
    
    if expected_ndim is not None: 
        if array.ndim != expected_ndim:
            raise ValueError(
                f"{param_name} must be {expected_ndim}D, got {array.ndim}D"
            )
    
    if expected_shape is not None: 
        if len(expected_shape) != array.ndim:
            raise ValueError(
                f"{param_name} shape mismatch: expected {len(expected_shape)}D, "
                f"got {array.ndim}D"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, array.shape)):
            if expected != -1 and expected != actual:
                raise ValueError(
                    f"{param_name} dimension {i}:  expected {expected}, got {actual}"
                )
    
    return True


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def validate_split_ratios(train:  float,
                         val: float,
                         test: float,
                         tolerance: float = 0.01) -> bool:
    """
    Validate that train/val/test split ratios sum to 1.0.
    
    Args:
        train: Train ratio
        val: Validation ratio
        test: Test ratio
        tolerance: Allowed deviation from 1.0 (for floating point errors)
        
    Returns: 
        True if valid
        
    Raises:
        ValueError: If ratios don't sum to 1.0
        
    Example:
        >>> validate_split_ratios(0.6, 0.2, 0.2)
        True
    """
    total = train + val + test
    
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {total}). "
            f"train={train}, val={val}, test={test}"
        )
    
    # Check individual ratios are valid
    for ratio, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        validate_range(ratio, 0.0, 1.0, name, inclusive=True)
    
    return True


def validate_dataset_shape(X: np.ndarray,
                          y: np.ndarray,
                          require_2d: bool = True) -> bool:
    """
    Validate dataset X and y have compatible shapes.
    
    Args:
        X: Feature array
        y: Label array
        require_2d:  If True, require X to be 2D
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If shapes are incompatible
        
    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = np.array([0, 1])
        >>> validate_dataset_shape(X, y)
        True
    """
    if require_2d:
        validate_array_shape(X, expected_ndim=2, param_name="X")
    
    validate_array_shape(y, expected_ndim=1, param_name="y")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples. "
            f"X:  {X.shape[0]}, y: {y. shape[0]}"
        )
    
    return True


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config_key(config: dict,
                       key: str,
                       required: bool = True,
                       expected_type: Optional[type] = None) -> bool:
    """
    Validate that a configuration key exists and has correct type.
    
    Args:
        config: Configuration dictionary
        key: Key to validate
        required: If True, raise error if key is missing
        expected_type: Expected type of value (None = no type check)
        
    Returns: 
        True if valid
        
    Raises:
        KeyError:  If required key is missing
        TypeError: If value has wrong type
        
    Example: 
        >>> config = {'dataset':  'SECOM', 'model': 'neural_network'}
        >>> validate_config_key(config, 'dataset', expected_type=str)
        True
    """
    if key not in config:
        if required: 
            raise KeyError(f"Missing required config key: '{key}'")
        else:
            return False
    
    if expected_type is not None: 
        validate_type(config[key], expected_type, param_name=f"config['{key}']")
    
    return True


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == '__main__':
    from utils.colors import print_header, print_section, print_success
    
    print_header("VALIDATOR UTILITIES DEMO")
    
    # Test 1: File validation
    print_section("File Validation")
    
    try:
        # Create a test file
        Path("test_file.csv").touch()
        validate_file_path("test_file.csv", allowed_extensions=['. csv'])
        print_success("File validation passed")
        
        # Clean up
        Path("test_file.csv").unlink()
    except Exception as e:
        print_error(f"File validation failed:  {e}")
    
    # Test 2: Numeric range validation
    print("\n")
    print_section("Numeric Range Validation")
    
    try:
        validate_range(0.5, 0.0, 1.0, "dropout")
        print_success("Range validation passed (0.5 in [0.0, 1.0])")
    except Exception as e:
        print_error(f"Range validation failed: {e}")
    
    try:
        validate_range(1.5, 0.0, 1.0, "dropout")
        print_error("Should have raised ValueError")
    except ValueError as e:
        print_success(f"Correctly caught out-of-range:  {e}")
    
    # Test 3: Choice validation
    print("\n")
    print_section("Choice Validation")
    
    try:
        validate_choice("relu", ["relu", "tanh", "elu"], "activation")
        print_success("Choice validation passed ('relu' in choices)")
    except Exception as e: 
        print_error(f"Choice validation failed: {e}")
    
    try:
        validate_choice("sigmoid", ["relu", "tanh", "elu"], "activation")
        print_error("Should have raised ValueError")
    except ValueError as e:
        print_success(f"Correctly caught invalid choice: {e}")
    
    # Test 4: Type validation
    print("\n")
    print_section("Type Validation")
    
    try:
        validate_type(5, int, "population_size")
        print_success("Type validation passed (5 is int)")
    except Exception as e:
        print_error(f"Type validation failed: {e}")
    
    try:
        validate_type(0.5, (int, float), "learning_rate")
        print_success("Type validation passed (0.5 is int or float)")
    except Exception as e:
        print_error(f"Type validation failed: {e}")
    
    # Test 5: Array validation
    print("\n")
    print_section("Array Validation")
    
    try:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        validate_dataset_shape(X, y)
        print_success("Dataset shape validation passed")
    except Exception as e:
        print_error(f"Dataset validation failed: {e}")
    
    # Test 6: Split ratios
    print("\n")
    print_section("Split Ratio Validation")
    
    try:
        validate_split_ratios(0.6, 0.2, 0.2)
        print_success("Split ratios valid (0.6 + 0.2 + 0.2 = 1.0)")
    except Exception as e:
        print_error(f"Split ratio validation failed: {e}")
    
    try:
        validate_split_ratios(0.6, 0.3, 0.3)
        print_error("Should have raised ValueError")
    except ValueError as e:
        print_success(f"Correctly caught invalid split:  {e}")
    
    print("\n")
    print_header("DEMO COMPLETE")