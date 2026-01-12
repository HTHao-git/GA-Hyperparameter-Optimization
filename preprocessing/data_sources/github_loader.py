# ============================================================================
# GITHUB DATASETS LOADER  
# ============================================================================
# Downloads datasets from GitHub repositories
#
# Last updated: 2025-12-31
# ============================================================================

import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from utils.logger import Logger


class GitHubLoader: 
    """Loader for datasets hosted on GitHub."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def download(self,
                url: str,
                destination:  Path,
                show_progress:  bool = True) -> bool:
        """
        Download file from GitHub. 
        
        Args:
            url: Full URL to file (raw.githubusercontent.com)
            destination: Local path to save file
            show_progress: Show progress bar
            
        Returns: 
            True if successful
        """
        try:
            if self.logger:
                self.logger.info(f"Downloading from GitHub: {url}")
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
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
                self.logger.success(f"Downloaded: {destination.name}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"GitHub download failed: {e}")
            return False