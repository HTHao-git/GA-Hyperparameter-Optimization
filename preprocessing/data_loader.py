# ============================================================================
# DATA LOADER - Multi-Source Dataset Loading System
# ============================================================================
# Universal dataset loader with automatic download and manual upload support. 
#
# FEATURES:
#   - Automatic download from multiple sources (UCI, OpenML, GitHub, TensorFlow)
#   - Manual upload support for custom datasets
#   - File format auto-detection
#   - Interactive prompts with file browser
#   - Caching and metadata tracking
#
# USAGE:
#   from preprocessing.data_loader import DatasetLoader
#   
#   loader = DatasetLoader()
#   X, y, metadata = loader.load_dataset('secom')
#
# Last updated: 2025-12-31
# ============================================================================

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

from utils.logger import get_logger, Logger
from utils.validators import validate_file_path, validate_directory
from utils.colors import (
    print_header, print_section, print_success, print_error, 
    print_warning, print_info, Colors, colored
)

from preprocessing.data_sources.uci_loader import UCILoader
from preprocessing.data_sources.tensorflow_loader import TensorFlowLoader
from preprocessing.data_sources.github_loader import GitHubLoader
from preprocessing.data_sources.kaggle_loader import KaggleLoader
from preprocessing.data_sources. uciml_loader import UCIMLRepoLoader
from preprocessing.data_sources.openml_loader import OpenMLLoader

from preprocessing.file_parsers.auto_parser import AutoParser
from preprocessing.file_parsers.space_separated_parser import SpaceSeparatedParser


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DatasetNotFoundError(Exception):
    """Raised when dataset cannot be found or downloaded."""
    pass


class DatasetParsingError(Exception):
    """Raised when dataset parsing fails."""
    pass


# ============================================================================
# DATASET LOADER CLASS
# ============================================================================

class DatasetLoader:
    """
    Universal dataset loader with multi-source support.
    
    Args:
        dataset_dir: Base directory for datasets (default: 'datasets/')
        registry_path: Path to datasets registry JSON
        logger: Logger instance (optional)
        interactive: Enable interactive prompts (default: True)
    """
    
    def __init__(self,
                 dataset_dir: str = "datasets",
                 registry_path: str = "config/datasets_registry.json",
                 logger: Optional[Logger] = None,
                 interactive: bool = True):
        
        self.dataset_dir = Path(dataset_dir)
        self.registry_path = Path(registry_path)
        self.interactive = interactive
        
        # Logger
        if logger is None: 
            self.logger = get_logger(name="DataLoader", verbose=True)
        else:
            self.logger = logger
        
        # Create dataset directory
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Load registry
        self.registry = self._load_registry()
        
        # Initialize source loaders
        self.uci_loader = UCILoader(self.logger)
        self.tf_loader = TensorFlowLoader(self.logger)
        self.github_loader = GitHubLoader(self.logger)
        self.kaggle_loader = KaggleLoader(self.logger)
        self.uciml_loader = UCIMLRepoLoader(self.logger)
        self.openml_loader = OpenMLLoader(self.logger)
        
        # Initialize parsers
        self.auto_parser = AutoParser(self.logger)
        self.space_parser = SpaceSeparatedParser(self.logger)
    
    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    
    def load_dataset(self,
                    name: str,
                    source:  str = 'auto',
                    force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset from best available source.
        
        Args:
            name: Dataset name ('isolet', 'secom', 'custom/my_data', etc.)
            source: 'auto', 'uci', 'tensorflow', 'github', 'kaggle', 'local'
            force_download: Force re-download even if cached
            
        Returns:
            (X, y, metadata) tuple
            
        Raises:
            DatasetNotFoundError: If dataset cannot be loaded
        """
        print_header(f"LOADING DATASET: {name}")
        
        # Check if dataset exists locally
        dataset_path = self.dataset_dir / name
        
        if not force_download and self._exists_locally(name):
            self.logger.info(f"Found cached dataset: {dataset_path}")
            return self._load_local(name)
        
        # Not cached - try to download or prompt for manual upload
        self.logger.warning("Dataset not found locally")
        self.logger.blank()
        
        # Check if it's a built-in dataset
        if name in self.registry:
            return self._load_builtin(name, source, force_download)
        else:
            return self._load_custom(name)
    
    # ========================================================================
    # BUILT-IN DATASETS
    # ========================================================================
    
    def _load_builtin(self,
                     name: str,
                     source: str,
                     force_download: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load built-in dataset with automatic download."""
        
        dataset_config = self.registry[name]
        dataset_path = self.dataset_dir / name
        
        # Try automatic download
        if source == 'auto':
            success = self._auto_download(name, dataset_config, dataset_path)
        else:
            success = self._download_from_source(name, dataset_config, dataset_path, source)
        
        if success:
            # Download successful, now parse
            return self._parse_builtin(name, dataset_config, dataset_path)
        else:
            # Download failed, prompt for manual upload
            if self.interactive:
                return self._prompt_manual_upload(name, dataset_config, dataset_path)
            else:
                raise DatasetNotFoundError(
                    f"Dataset '{name}' could not be downloaded and interactive mode is disabled"
                )
    
    def _auto_download(self,
                      name: str,
                      dataset_config: Dict,
                      dataset_path: Path) -> bool:
        """Try downloading from all available sources."""
        
        print_section("AUTOMATIC DOWNLOAD OPTIONS")
        print()
        
        # Get available sources sorted by priority
        sources = dataset_config['sources']
        sorted_sources = sorted(
            sources.items(),
            key=lambda x: x[1].get('priority', 99)
        )
        
        # Show available sources
        print_info(f"Available sources for '{name}':")
        for i, (source_name, source_config) in enumerate(sorted_sources, 1):
            note = source_config.get('note', '')
            if note:
                print(f"  {i}.{source_name} ({note})")
            else:
                print(f"  {i}.{source_name}")
        print()
        
        # Ask user if they want to attempt download
        if self.interactive:
            response = input("Attempt automatic download? (y/n): ").strip().lower()
            print()
            
            if response != 'y':
                return False
        
        # Try each source
        for i, (source_name, source_config) in enumerate(sorted_sources, 1):
            self.logger.info(f"Trying source {i}/{len(sorted_sources)}: {source_name}...")
            
            try:
                success = self._download_from_source(
                    name, 
                    dataset_config, 
                    dataset_path, 
                    source_name
                )
                
                if success: 
                    self.logger.success(f"Downloaded from {source_name}")
                    return True
                    
            except Exception as e: 
                self.logger.error(f"Download failed:  {e}")
                continue
        
        # All sources failed
        self.logger.error("All automatic download attempts failed")
        return False
    
    def _download_from_source(self,
                             name: str,
                             dataset_config: Dict,
                             dataset_path: Path,
                             source: str) -> bool:
        """Download from specific source."""
        
        if source not in dataset_config['sources']: 
            self.logger.error(f"Source '{source}' not available for '{name}'")
            return False
        
        source_config = dataset_config['sources'][source]
        
        # Route to appropriate loader
        if source == 'uci': 
            return self.uci_loader.download_dataset(source_config, dataset_path)
        
        elif source == 'tensorflow': 
            return self.tf_loader.load_fashion_mnist(dataset_path)
        
        elif source == 'github':
            base_url = source_config['url']
            files = source_config['files']
            
            success = True
            for file_key, filename in files.items():
                url = base_url + filename
                destination = dataset_path / filename
                
                if not self.github_loader.download(url, destination):
                    success = False
                    break
            
            return success
        
        elif source == 'kaggle': 
            if not self.kaggle_loader.is_available():
                if self.interactive:
                    # Prompt to set up Kaggle
                    self.kaggle_loader.prompt_setup()
                    
                    if not self.kaggle_loader.is_available():
                        return False
                else:
                    return False
            
            dataset_id = source_config.get('dataset_id')
            if dataset_id:
                return self.kaggle_loader.download_dataset(dataset_id, dataset_path)
            else:
                return False

        elif source == 'openml':
            dataset_id = source_config.get('dataset_id')
            if dataset_id:
                return self.openml_loader.download_dataset(dataset_id, dataset_path)
            else:
                return False
        
        elif source == 'uciml':
            dataset_id = source_config. get('dataset_id')
            if dataset_id:
                dataset_name = name
                return self.uciml_loader.download_dataset(dataset_id, dataset_path, dataset_name)
            else:
                return False

        else:
            self.logger.error(f"Unknown source: {source}")
            return False
    

    # ========================================================================
    # PARSING
    # ========================================================================
    
    def _parse_builtin(self,
                      name: str,
                      dataset_config:  Dict,
                      dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Parse built-in dataset based on its format."""
        
        self.logger.info("Parsing dataset files...")
        
        # Special handling for specific datasets
        if name == 'secom':
            return self._parse_secom(dataset_path)
        
        elif name == 'fashion_mnist':
            return self._parse_fashion_mnist(dataset_path)
        
        elif name == 'isolet':
            return self._parse_isolet(dataset_path)
        
        elif name == 'steel_plates':
            return self._parse_steel_plates(dataset_path)
        
        else:
            # Generic parsing
            return self._parse_generic(dataset_path)
    
    def _parse_secom(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Parse SECOM dataset (separate features and labels files)."""
        
        features_file = dataset_path / "secom.data"
        labels_file = dataset_path / "secom_labels.data"
        
        # Check if files exist
        if not features_file.exists():
            # Maybe it's a CSV from GitHub
            csv_file = dataset_path / "secom.csv"
            if csv_file.exists():
                return self.auto_parser.parse(csv_file, has_header=True)
            else:
                raise DatasetNotFoundError(f"SECOM data files not found in {dataset_path}")
        
        # Parse using space-separated parser
        X, y, metadata = self.space_parser.parse_secom(features_file, labels_file)
        
        # Save metadata
        self._save_metadata(dataset_path, metadata)
        
        return X, y, metadata
    
    def _parse_fashion_mnist(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Parse Fashion-MNIST dataset."""
        
        npz_file = dataset_path / "fashion_mnist.npz"
        
        if npz_file.exists():
            return self.auto_parser.parse(npz_file, format_hint='npz')
        else:
            raise DatasetNotFoundError(f"Fashion-MNIST file not found:  {npz_file}")
    
    def _parse_isolet(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Parse ISOLET dataset."""
        
        train_file = dataset_path / "isolet1+2+3+4.data"
        test_file = dataset_path / "isolet5.data"
        
        if not train_file.exists():
            # Try alternative names
            alternatives = list(dataset_path.glob("*.data"))
            if len(alternatives) >= 2:
                train_file = alternatives[0]
                test_file = alternatives[1]
            else:
                raise DatasetNotFoundError(f"ISOLET data files not found in {dataset_path}")
        
        # Parse train and test files
        X_train, y_train, _ = self.auto_parser.parse(train_file, format_hint='space_separated', target_column=-1)
        X_test, y_test, _ = self.auto_parser.parse(test_file, format_hint='space_separated', target_column=-1)
        
        # Combine
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        
        # ISOLET labels are 1-26 (A-Z), convert to 0-25
        y = y - 1
        
        metadata = {
            'filepath_train': str(train_file),
            'filepath_test': str(test_file),
            'format': 'isolet',
            'samples': X.shape[0],
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features':  X.shape[1],
            'classes': len(np.unique(y)),
            'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        }
        
        self._save_metadata(dataset_path, metadata)
        
        return X, y, metadata
    
    def _parse_steel_plates(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Parse Steel Plates dataset."""
        
        # Try NNA format
        nna_file = dataset_path / "Faults.NNA"
        
        if nna_file.exists():
            X, y, metadata = self.auto_parser.parse(nna_file, format_hint='nna')
            self._save_metadata(dataset_path, metadata)
            return X, y, metadata
        
        # Try CSV format (from Kaggle)
        csv_file = dataset_path / "faults.csv"
        
        if csv_file.exists():
            X, y, metadata = self.auto_parser.parse(csv_file, format_hint='csv', has_header=True)
            self._save_metadata(dataset_path, metadata)
            return X, y, metadata
        
        raise DatasetNotFoundError(f"Steel Plates data files not found in {dataset_path}")
    
    def _parse_generic(self, dataset_path:  Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Generic parsing for unknown datasets."""
        
        # Find first data file
        data_files = list(dataset_path.glob("*.csv")) + \
                    list(dataset_path.glob("*.data")) + \
                    list(dataset_path.glob("*.txt"))
        
        if not data_files:
            raise DatasetNotFoundError(f"No data files found in {dataset_path}")
        
        data_file = data_files[0]
        
        return self.auto_parser.parse(data_file)
    
    # ========================================================================
    # CUSTOM DATASETS
    # ========================================================================
    
    def _load_custom(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load custom dataset (user-provided)."""
        
        dataset_path = self.dataset_dir / name
        
        print_section("CUSTOM DATASET SETUP")
        print()
        print_info(f"Custom dataset '{name}' not found.")
        print()
        
        if not self.interactive:
            raise DatasetNotFoundError(
                f"Custom dataset '{name}' not found and interactive mode is disabled"
            )
        
        # Prompt user for file
        print("How would you like to add your dataset? ")
        print("  1. I've already placed files in datasets/{}".format(name))
        print("  2. Browse for file on my computer")
        print("  3. Enter file path manually")
        print()
        
        choice = input("Choose option (1-3): ").strip()
        print()
        
        if choice == '1':
            return self._load_existing_custom(dataset_path)
        
        elif choice == '2': 
            file_path = self._browse_for_file()
            if file_path: 
                return self._load_custom_file(file_path, dataset_path, name)
            else:
                raise DatasetNotFoundError("No file selected")
        
        elif choice == '3':
            file_path_str = input("Enter file path:  ").strip()
            if file_path_str:
                file_path = Path(file_path_str)
                if file_path.exists():
                    return self._load_custom_file(file_path, dataset_path, name)
                else:
                    raise DatasetNotFoundError(f"File not found: {file_path}")
            else:
                raise DatasetNotFoundError("No file path provided")
        
        else: 
            raise DatasetNotFoundError("Invalid option")
    
    def _load_existing_custom(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load custom dataset from existing directory."""
        
        if not dataset_path.exists():
            raise DatasetNotFoundError(f"Directory not found: {dataset_path}")
        
        # Find data files
        data_files = list(dataset_path.glob("*.csv")) + \
                    list(dataset_path.glob("*.data")) + \
                    list(dataset_path.glob("*.txt")) + \
                    list(dataset_path.glob("*.xlsx"))
        
        if not data_files:
            raise DatasetNotFoundError(f"No data files found in {dataset_path}")
        
        # Use first file
        data_file = data_files[0]
        
        self.logger.info(f"Found data file: {data_file.name}")
        
        return self.auto_parser.parse(data_file)
    
    def _load_custom_file(self,
                         file_path: Path,
                         dataset_path:  Path,
                         name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load custom dataset from specific file."""
        
        import shutil
        
        self.logger.info(f"Loading custom dataset: {file_path.name}")
        
        # Create dataset directory
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Copy file to dataset directory
        destination = dataset_path / file_path.name
        shutil.copy(file_path, destination)
        
        self.logger.success(f"Copied to:  {destination}")
        
        # Parse file
        X, y, metadata = self.auto_parser.parse(destination)
        
        # Save metadata
        self._save_metadata(dataset_path, metadata)
        
        return X, y, metadata
    
    # ========================================================================
    # INTERACTIVE PROMPTS
    # ========================================================================
    
    def _prompt_manual_upload(self,
                             name: str,
                             dataset_config: Dict,
                             dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        """Prompt user to manually upload dataset."""
        
        print_section("ALL AUTOMATIC DOWNLOADS FAILED")
        print()
        print_info("You can manually download the dataset:")
        print()
        
        # Show download instructions
        sources = dataset_config['sources']
        
        option_num = 1
        
        if 'uci' in sources:
            print(colored(f"  Option {option_num}: Download from UCI", Colors.INFO, bold=True))
            print(f"    URL: {sources['uci']['url']}")
            print(f"    Save to: {dataset_path}/")
            print()
            option_num += 1
        
        if 'kaggle' in sources:
            print(colored(f"  Option {option_num}: Download from Kaggle", Colors.INFO, bold=True))
            dataset_id = sources['kaggle'].get('dataset_id', 'N/A')
            print(f"    URL: https://www.kaggle.com/datasets/{dataset_id}")
            print(f"    Instructions: Set up Kaggle API (see above)")
            print()
            option_num += 1
        
        print(colored(f"  Option {option_num}: Use your own dataset file", Colors.INFO, bold=True))
        print(f"    Place file in: {dataset_path}/")
        print(f"    Supported formats: .csv, .data, .NNA, .txt, .xlsx")
        print()
        
        print("─" * 70)
        print()
        print("What would you like to do?")
        print("  1. I've placed the file manually (check now)")
        print("  2. Browse for file on my computer")
        print("  3. Enter custom file path")
        print("  4. Skip this dataset for now")
        print()
        
        choice = input("Choose option (1-4): ").strip()
        print()
        
        if choice == '1': 
            return self._load_existing_custom(dataset_path)
        
        elif choice == '2': 
            file_path = self._browse_for_file()
            if file_path:
                return self._load_custom_file(file_path, dataset_path, name)
            else:
                raise DatasetNotFoundError("No file selected")
        
        elif choice == '3':
            file_path_str = input("Enter file path: ").strip()
            if file_path_str: 
                file_path = Path(file_path_str)
                if file_path.exists():
                    return self._load_custom_file(file_path, dataset_path, name)
                else:
                    raise DatasetNotFoundError(f"File not found: {file_path}")
            else:
                raise DatasetNotFoundError("No file path provided")
        
        elif choice == '4':
            raise DatasetNotFoundError(f"Dataset '{name}' skipped by user")
        
        else: 
            print_warning("Invalid choice")
            raise DatasetNotFoundError("Invalid option")
    
    def _browse_for_file(self) -> Optional[Path]:
        """Open GUI file browser."""
        
        try:
            from tkinter import Tk, filedialog
            
            self.logger.info("Opening file browser...")
            
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select dataset file",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Data files", "*.data"),
                    ("Text files", "*.txt"),
                    ("Excel files", "*.xlsx"),
                    ("NNA files", "*.NNA"),
                    ("All files", "*.*")
                ]
            )
            
            root.destroy()
            
            if file_path:
                self.logger.success(f"File selected: {Path(file_path).name}")
                return Path(file_path)
            else:
                self.logger.warning("No file selected")
                return None
        
        except ImportError:
            self.logger.warning("GUI file browser not available (tkinter not installed)")
            return None
        except Exception as e:
            self.logger.error(f"File browser failed: {e}")
            return None
    
    # ========================================================================
    # LOCAL CACHE
    # ========================================================================
    
    def _exists_locally(self, name: str) -> bool:
        """Check if dataset exists in local cache."""
        
        dataset_path = self.dataset_dir / name
        
        if not dataset_path.exists():
            return False
        
        # Check if directory has data files
        data_files = list(dataset_path.glob("*.csv")) + \
                    list(dataset_path.glob("*.data")) + \
                    list(dataset_path.glob("*.npz")) + \
                    list(dataset_path.glob("*.NNA"))
        
        return len(data_files) > 0
    
    def _load_local(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load dataset from local cache."""
        
        dataset_path = self.dataset_dir / name
        
        # Check if it's a built-in dataset
        if name in self.registry:
            dataset_config = self.registry[name]
            return self._parse_builtin(name, dataset_config, dataset_path)
        else:
            # Custom dataset
            return self._parse_generic(dataset_path)
    
    # ========================================================================
    # METADATA
    # ========================================================================
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry from JSON file."""
        
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            self.logger.debug(f"Loaded registry:  {len(registry)} datasets")
            
            return registry
        
        except FileNotFoundError:
            self.logger.warning(f"Registry not found:  {self.registry_path}")
            return {}
        except json.JSONDecodeError as e: 
            self.logger.error(f"Invalid registry JSON: {e}")
            return {}
    
    def _save_metadata(self, dataset_path:  Path, metadata: Dict[str, Any]):
        """Save dataset metadata to JSON file."""
        
        metadata_file = dataset_path / "metadata.json"
        
        # Add timestamp
        metadata['loaded_at'] = datetime.now().isoformat()
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Saved metadata: {metadata_file}")
        
        except Exception as e:
            self.logger.warning(f"Failed to save metadata:  {e}")
    
    def _load_metadata(self, dataset_path: Path) -> Optional[Dict[str, Any]]: 
        """Load dataset metadata from JSON file."""
        
        metadata_file = dataset_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return metadata
        
        except Exception as e:
            self.logger.warning(f"Failed to load metadata:  {e}")
            return None
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def list_available_datasets(self) -> List[str]:
        """Get list of available built-in datasets."""
        return list(self.registry.keys())
    
    def list_cached_datasets(self) -> List[str]:
        """Get list of datasets in local cache."""
        
        cached = []
        
        for item in self.dataset_dir.iterdir():
            if item.is_dir() and self._exists_locally(item.name):
                cached.append(item.name)
        
        return cached
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        
        if name in self.registry:
            return self.registry[name]
        else:
            # Try to load metadata from cache
            dataset_path = self.dataset_dir / name
            return self._load_metadata(dataset_path) or {}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
    """
    Convenience function to load a dataset.
    
    Args:
        name: Dataset name
        **kwargs: Additional arguments for DatasetLoader
        
    Returns:
        (X, y, metadata) tuple
    """
    loader = DatasetLoader(**kwargs)
    return loader.load_dataset(name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Test the loader
    loader = DatasetLoader()
    
    print_header("DATA LOADER TEST")
    
    print("\nAvailable datasets:")
    for dataset in loader.list_available_datasets():
        print(f"  - {dataset}")
    
    print("\nCached datasets:")
    for dataset in loader.list_cached_datasets():
        print(f"  - {dataset}")
    
    print("\n" + "="*70)
    print("Ready to load datasets!")
    print("="*70)