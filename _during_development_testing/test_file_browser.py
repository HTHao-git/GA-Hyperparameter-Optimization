# Test the tkinter file browser

from preprocessing.data_loader import DatasetLoader
from utils.logger import get_logger

logger = get_logger(name="BROWSER_TEST", verbose=True)

loader = DatasetLoader(logger=logger, interactive=True)

print("="*70)
print("FILE BROWSER TEST")
print("="*70)
print()
print("This will open a file browser dialog.")
print("Select any data file (. csv, .txt, .data) to test.")
print()
input("Press Enter to open file browser...")

file_path = loader._browse_for_file()

if file_path:
    print()
    print(f"✓ Selected: {file_path}")
    print(f"  Name: {file_path.name}")
    print(f"  Extension: {file_path.suffix}")
    print(f"  Exists: {file_path.exists()}")
else:
    print()
    print("No file selected")