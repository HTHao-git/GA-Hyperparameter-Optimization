# Test logger with file output
from utils.logger import get_logger
from pathlib import Path

# Create logger with file output
log_file = "outputs/test_log.txt"
Path("outputs").mkdir(exist_ok=True)

logger = get_logger(name="TEST", log_file=log_file, verbose=True)

logger.header("TESTING LOGGER WITH FILE OUTPUT")

logger.section("Basic Logging")
logger.success("Success message test")
logger.error("Error message test")
logger.warning("Warning message test")
logger.info("Info message test")
logger.debug("Debug message test")

logger.blank()
logger.section("Progress Tracking")
for i in range(1, 6):
    logger.progress(i, 5, "Processing batch")

logger.blank()
logger.section("Custom Formatting")
logger.separator()
logger.info("This is important information")
logger.separator('=')

logger.blank()
logger.header("TEST COMPLETE")

print("\n" + "="*70)
from utils.colors import print_success, print_info
print_success(f"Log file created: {log_file}")
print_info("Check the file to verify both console and file logging work")
print("="*70)

# Display log file contents
print("\nLog file contents:")
print("-"*70)
with open(log_file, 'r') as f:
    print(f.read())