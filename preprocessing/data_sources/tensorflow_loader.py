# ============================================================================
# TENSORFLOW/KERAS DATASETS LOADER
# ============================================================================
# Loads datasets from TensorFlow/Keras built-in datasets
#
# Last updated: 2025-12-31
# ============================================================================

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.logger import Logger


class TensorFlowLoader: 
    """Loader for TensorFlow/Keras built-in datasets."""
    
    def __init__(self, logger:  Optional[Logger] = None):
        self.logger = logger
    
    def load_fashion_mnist(self, 
                          destination_dir: Path) -> bool:
        """
        Load Fashion-MNIST using TensorFlow/Keras. 
        
        Args:
            destination_dir: Directory to save dataset
            
        Returns:
            True if successful
        """
        try:
            import tensorflow as tf
            
            if self.logger:
                self. logger.info("Downloading Fashion-MNIST via TensorFlow...")
            
            # Download using Keras
            (X_train, y_train), (X_test, y_test) = \
                tf.keras.datasets. fashion_mnist.load_data()
            
            # Create destination directory
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as . npz file
            save_path = destination_dir / "fashion_mnist.npz"
            np.savez_compressed(
                save_path,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            if self.logger:
                self.logger.success(f"Saved to: {save_path}")
                self.logger.info(f"  Train:  {X_train.shape[0]} samples")
                self.logger.info(f"  Test: {X_test.shape[0]} samples")
            
            return True
            
        except Exception as e:
            if self. logger:
                self.logger. error(f"TensorFlow download failed: {e}")
            return False