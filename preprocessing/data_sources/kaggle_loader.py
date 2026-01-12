# ============================================================================
# KAGGLE DATASETS LOADER
# ============================================================================
# Downloads datasets from Kaggle (supports both legacy and new auth methods)
#
# AUTHENTICATION METHODS:
#   1. Legacy: kaggle.json file in ~/.kaggle/
#   2. New: KAGGLE_API_TOKEN environment variable
#
# Last updated: 2025-12-31
# ============================================================================

import os
from pathlib import Path
from typing import Optional

from utils.logger import Logger
from utils.colors import print_warning, print_info, print_section, print_success


class KaggleLoader:
    """Loader for Kaggle datasets (supports legacy and new authentication)."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.api = None
        self.auth_method = None
        self._check_credentials()
    
    def _check_credentials(self) -> bool:
        """Check if Kaggle API credentials are configured."""
        
        # Method 1: Check for new token-based auth (environment variable)
        if self._check_token_auth():
            self.auth_method = 'token'
            return True
        
        # Method 2: Check for legacy kaggle.json file
        if self._check_json_auth():
            self.auth_method = 'json'
            return True
        
        # No credentials found
        self.auth_method = None
        if self.logger:
            self.logger.warning("Kaggle API not configured")
        
        return False
    
    def _check_token_auth(self) -> bool:
        """Check for new token-based authentication."""
        
        token = os.environ.get('KAGGLE_API_TOKEN')
        
        if token:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                
                # Set the token as environment variable for the API
                os.environ['KAGGLE_KEY'] = token
                os.environ['KAGGLE_USERNAME'] = 'token'  # Placeholder username
                
                self.api = KaggleApi()
                self.api.authenticate()
                
                if self.logger:
                    self.logger.success("Kaggle API authenticated (token method)")
                
                return True
                
            except Exception as e: 
                if self.logger:
                    self.logger.debug(f"Token auth failed: {e}")
                return False
        
        return False
    
    def _check_json_auth(self) -> bool:
        """Check for legacy kaggle.json authentication."""
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            self.api = KaggleApi()
            self.api.authenticate()
            
            if self.logger:
                self.logger.success("Kaggle API authenticated (kaggle.json method)")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"JSON auth failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Kaggle API is available."""
        return self.api is not None
    
    def download_dataset(self,
                        dataset_id: str,
                        destination_dir: Path,
                        unzip: bool = True) -> bool:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_id: Kaggle dataset ID (e.g., 'uciml/faulty-steel-plates')
            destination_dir: Directory to save dataset
            unzip:  Automatically unzip downloaded files
            
        Returns: 
            True if successful
        """
        if not self.is_available():
            if self.logger:
                self.logger.error("Kaggle API not available")
            return False
        
        try:
            if self.logger:
                self.logger.info(f"Downloading from Kaggle: {dataset_id}")
            
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            self.api.dataset_download_files(
                dataset_id,
                path=str(destination_dir),
                unzip=unzip
            )
            
            if self.logger:
                self.logger.success(f"Downloaded to: {destination_dir}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Kaggle download failed: {e}")
            return False
    
    def prompt_setup(self) -> bool:
        """
        Interactive prompt to help user set up Kaggle API. 
        
        Returns:
            True if setup successful
        """
        print_section("KAGGLE SETUP REQUIRED")
        print()
        print_info("Some datasets are available on Kaggle with better formatting.")
        print_warning("Kaggle API credentials not found.")
        print()
        print("Kaggle offers TWO authentication methods:")
        print()
        
        print("─" * 70)
        print("METHOD 1: New Token-Based Authentication (Recommended)")
        print("─" * 70)
        print()
        print("  1. Go to:  https://www.kaggle.com/")
        print("  2. Login → Settings → API → Create New Token")
        print("  3. Copy the provided token")
        print("  4. Set environment variable:")
        print()
        print("     Windows (PowerShell):")
        print("       $env:KAGGLE_API_TOKEN = \"your_token_here\"")
        print()
        print("     Windows (Command Prompt):")
        print("       set KAGGLE_API_TOKEN=your_token_here")
        print()
        print("     macOS/Linux:")
        print("       export KAGGLE_API_TOKEN=\"your_token_here\"")
        print()
        print("  Note: This is temporary. To make it permanent:")
        print("    - Windows: Add to System Environment Variables")
        print("    - macOS/Linux: Add to ~/.bashrc or ~/.zshrc")
        print()
        
        print("─" * 70)
        print("METHOD 2: Legacy kaggle.json File")
        print("─" * 70)
        print()
        print("  1. Go to: https://www.kaggle.com/")
        print("  2. Login → Settings → API")
        print("  3. Look for 'Legacy API Tokens'")
        print("  4. Click 'Create New Token' (legacy)")
        print("  5. Download kaggle.json file")
        
        # Determine default path
        home = Path.home()
        default_path = home / ".kaggle" / "kaggle.json"
        
        print(f"  6. Place in: {default_path}")
        print()
        
        print("─" * 70)
        print()
        
        # Ask which method user wants to use
        print("Which method would you like to use?")
        print("  1. Token-based (I have or will set KAGGLE_API_TOKEN)")
        print("  2. Legacy kaggle.json file")
        print("  3. Skip for now")
        print()
        
        choice = input("Choose option (1-3): ").strip()
        print()
        
        if choice == '1':
            return self._setup_token_auth()
        elif choice == '2':
            return self._setup_json_auth()
        else:
            print_info("Skipping Kaggle setup for now.")
            return False
    
    def _setup_token_auth(self) -> bool:
        """Setup token-based authentication."""
        
        print_info("Token-based authentication setup")
        print()
        print("Do you already have the KAGGLE_API_TOKEN set?  ")
        print("  1. Yes, I've already set it")
        print("  2. No, I need to enter it now")
        print()
        
        choice = input("Choose option (1-2): ").strip()
        print()
        
        if choice == '1': 
            # Try to re-check credentials
            if self._check_token_auth():
                print_success("✓ Token detected and working!")
                return True
            else:
                print_warning("Token not found. Make sure you've set KAGGLE_API_TOKEN.")
                print_info("You may need to restart your terminal/IDE for changes to take effect.")
                return False
        
        elif choice == '2': 
            print("Enter your Kaggle API token:")
            token = input("Token: ").strip()
            
            if token:
                # Set temporarily for this session
                os.environ['KAGGLE_API_TOKEN'] = token
                
                # Try to authenticate
                if self._check_token_auth():
                    print()
                    print_success("✓ Token authenticated successfully!")
                    print()
                    print_warning("Note: This is only set for the current session.")
                    print_info("To make it permanent, add to your environment variables:")
                    print()
                    print("  Windows (System Environment Variables):")
                    print("    1. Search 'Environment Variables' in Start Menu")
                    print("    2. Add new variable: KAGGLE_API_TOKEN = your_token")
                    print()
                    print("  Or add to your shell profile:")
                    print(f"    export KAGGLE_API_TOKEN=\"{token}\"")
                    print()
                    
                    return True
                else: 
                    print_warning("Token authentication failed.  Please check the token.")
                    return False
            else:
                print_warning("No token provided.")
                return False
        
        return False
    
    def _setup_json_auth(self) -> bool:
        """Setup legacy kaggle.json authentication."""
        
        print_info("Legacy kaggle.json authentication setup")
        print()
        
        response = input("Do you have kaggle.json file? (y/n): ").strip().lower()
        
        if response != 'y':
            print()
            print_info("Skipping Kaggle setup for now.")
            print_info("You can set it up later by following the steps above.")
            return False
        
        print()
        print("Where is your kaggle.json file?")
        
        home = Path.home()
        default_path = home / ".kaggle"
        
        print(f"  1. Default location ({default_path})")
        print("  2. Browse for file")
        print("  3. Enter path manually")
        print()
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1': 
            kaggle_path = default_path / "kaggle.json"
        elif choice == '2':
            kaggle_path = self._browse_for_kaggle_json()
        elif choice == '3':
            path_str = input("Enter path to kaggle.json: ").strip()
            kaggle_path = Path(path_str) if path_str else None
        else:
            print_warning("Invalid choice")
            return False
        
        if kaggle_path and kaggle_path.exists():
            return self._install_credentials(kaggle_path)
        else:
            print_warning(f"File not found:  {kaggle_path}")
            return False
    
    def _browse_for_kaggle_json(self) -> Optional[Path]:
        """Open file browser to select kaggle.json."""
        try:
            from tkinter import Tk, filedialog
            
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select kaggle.json file",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            root.destroy()
            
            return Path(file_path) if file_path else None
            
        except ImportError:
            print_warning("GUI file browser not available")
            return None
    
    def _install_credentials(self, source_path: Path) -> bool:
        """
        Install Kaggle credentials to default location.
        
        Args:
            source_path: Path to kaggle.json file
            
        Returns:
            True if successful
        """
        try:
            import shutil
            
            # Default location
            home = Path.home()
            kaggle_dir = home / ".kaggle"
            dest_path = kaggle_dir / "kaggle.json"
            
            # Create directory
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy(source_path, dest_path)
            
            # Set permissions (Kaggle requires 600 on Unix)
            if os.name != 'nt':  # Unix/Linux/macOS
                os.chmod(dest_path, 0o600)
            
            if self.logger:
                self.logger.success(f"Kaggle credentials installed:  {dest_path}")
            
            # Re-check credentials
            return self._check_credentials()
            
        except Exception as e: 
            if self.logger:
                self.logger.error(f"Failed to install credentials: {e}")
            return False