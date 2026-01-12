# ============================================================================
# UCI ML REPOSITORY LOADER
# ============================================================================
# Downloads datasets from UCI Machine Learning Repository
#
# Last updated: 2025-12-31
# ============================================================================

import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from utils.logger import Logger
from utils.colors import print_success, print_error, print_info


class UCILoader:
    """Loader for UCI ML Repository datasets."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def download(self, 
                url: str, 
                destination: Path,
                show_progress: bool = True) -> bool:
        """
        Download file from UCI repository.
        
        Args:
            url: Full URL to file
            destination: Local path to save file
            show_progress: Show progress bar
            
        Returns: 
            True if successful, False otherwise
        """
        try:
            if self.logger:
                self. logger.info(f"Downloading from UCI:  {url}")
            
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Write to file with progress bar
            with open(destination, 'wb') as f:
                if show_progress and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=destination.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            if self.logger:
                self.logger.success(f"Downloaded:  {destination. name}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"UCI download failed: {e}")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error:  {e}")
            return False
    
    def download_dataset(self,
                        dataset_config: Dict,
                        destination_dir: Path) -> bool:
        """
        Download all files for a dataset.
        
        Args:
            dataset_config:  Dataset configuration from registry
            destination_dir: Directory to save files
            
        Returns:
            True if all files downloaded successfully
        """
        base_url = dataset_config['url']
        files = dataset_config['files']
        
        success = True
        
        for file_key, filename in files.items():
            url = base_url + filename
            destination = destination_dir / filename
            
            # Skip if already exists
            if destination.exists():
                if self.logger:
                    self.logger.info(f"Already exists: {filename}")
                continue
            
            # Download
            if not self.download(url, destination):
                success = False
                break
        
        return success