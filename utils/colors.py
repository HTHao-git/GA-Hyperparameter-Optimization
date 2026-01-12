# ============================================================================
# COLORS - Universal Color System for Terminal Output
# ============================================================================
# This module provides a centralized color system for the entire project.
# All modules should import colors from here for consistency.
#
# USAGE:
#   from utils.colors import colored, Colors, print_colored
#   
#   print(colored("Success!", Colors.SUCCESS))
#   print_colored("Error!", Colors.ERROR)
#
# AVAILABLE COLORS:
#   - SUCCESS (green)
#   - ERROR (red)
#   - WARNING (yellow)
#   - INFO (cyan)
#   - DEBUG (magenta)
#   - HEADER (blue + bright)
#   - BOLD (bright white)
#   - DIM (dimmed)
#
# Last updated: 2025-12-31
# ============================================================================

from colorama import Fore, Back, Style, init
import sys

# Initialize colorama (required for Windows)
# autoreset=True means colors reset after each print
init(autoreset=True)


# ============================================================================
# COLOR CONSTANTS
# ============================================================================

class Colors:
    """
    Centralized color definitions for the entire project.
    
    Use these constants throughout the codebase for consistency.
    """
    
    # ========================================================================
    # SEMANTIC COLORS (use these for most cases)
    # ========================================================================
    
    SUCCESS = Fore.GREEN              # For success messages, checkmarks
    ERROR = Fore.RED                  # For errors, failures
    WARNING = Fore.YELLOW             # For warnings, alerts
    INFO = Fore.CYAN                  # For informational messages
    DEBUG = Fore.MAGENTA              # For debug output
    
    HEADER = Fore.BLUE + Style.BRIGHT # For section headers, titles
    BOLD = Style.BRIGHT               # For emphasis
    DIM = Style.DIM                   # For de-emphasized text
    
    # ========================================================================
    # RAW COLORS (for special cases)
    # ========================================================================
    
    BLACK = Fore.BLACK
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    
    # ========================================================================
    # BACKGROUND COLORS (rarely used, but available)
    # ========================================================================
    
    BG_BLACK = Back.BLACK
    BG_RED = Back.RED
    BG_GREEN = Back.GREEN
    BG_YELLOW = Back.YELLOW
    BG_BLUE = Back.BLUE
    BG_MAGENTA = Back.MAGENTA
    BG_CYAN = Back.CYAN
    BG_WHITE = Back.WHITE
    
    # ========================================================================
    # STYLES
    # ========================================================================
    
    BRIGHT = Style.BRIGHT
    NORMAL = Style.NORMAL
    RESET = Style.RESET_ALL


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def colored(text:  str, color: str = "", bold: bool = False) -> str:
    """
    Return colored text string.
    
    Args:
        text: Text to colorize
        color: Color code (use Colors.* constants)
        bold: If True, make text bold/bright
        
    Returns:
        Colored text string (includes reset at end)
        
    Example:
        >>> print(colored("Success!", Colors.SUCCESS))
        >>> print(colored("Error!", Colors.ERROR, bold=True))
    """
    if bold:
        return f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}"
    else:
        return f"{color}{text}{Style.RESET_ALL}"


def print_colored(text: str, color: str = "", bold: bool = False, end: str = '\n'):
    """
    Print colored text to console.
    
    Args:
        text: Text to print
        color: Color code (use Colors.* constants)
        bold: If True, make text bold/bright
        end: String appended after text (default:  newline)
        
    Example:
        >>> print_colored("Success!", Colors.SUCCESS)
        >>> print_colored("Loading...", Colors.INFO, end='')
    """
    print(colored(text, color, bold), end=end)


def print_header(text: str, char: str = '=', width: int = 70):
    """
    Print a formatted header with separator lines.
    
    Args:
        text: Header text
        char: Character for separator lines (default: '=')
        width: Total width of header (default: 70)
        
    Example: 
        >>> print_header("PHASE 1: DATA LOADING")
        ======================================================================
        PHASE 1: DATA LOADING
        ======================================================================
    """
    separator = char * width
    print_colored(separator, Colors.HEADER)
    print_colored(text, Colors.HEADER, bold=True)
    print_colored(separator, Colors.HEADER)


def print_section(text: str, char: str = '-', width: int = 70):
    """
    Print a formatted section divider (single line).
    
    Args:
        text: Section text
        char: Character for separator line (default: '-')
        width: Total width (default: 70)
        
    Example:
        >>> print_section("Configuration")
        ----------------------------------------------------------------------
        Configuration
    """
    separator = char * width
    print_colored(separator, Colors.INFO)
    print_colored(text, Colors.INFO, bold=True)


def print_success(text: str, prefix: str = "[SUCCESS]"):
    """
    Print success message in green.
    
    Args:
        text: Message text
        prefix: Prefix before message (default: "[SUCCESS]")
        
    Example:
        >>> print_success("Configuration loaded")
        [SUCCESS] Configuration loaded
    """
    print_colored(f"{prefix} {text}", Colors.SUCCESS)


def print_error(text: str, prefix: str = "[ERROR]"):
    """
    Print error message in red.
    
    Args:
        text: Message text
        prefix:  Prefix before message (default: "[ERROR]")
        
    Example: 
        >>> print_error("File not found")
        [ERROR] File not found
    """
    print_colored(f"{prefix} {text}", Colors.ERROR, bold=True)


def print_warning(text: str, prefix:  str = "[WARNING]"):
    """
    Print warning message in yellow.
    
    Args:
        text: Message text
        prefix: Prefix before message (default:  "[WARNING]")
        
    Example:
        >>> print_warning("Using default parameters")
        [WARNING] Using default parameters
    """
    print_colored(f"{prefix} {text}", Colors.WARNING)


def print_info(text: str, prefix: str = "[INFO]"):
    """
    Print info message in cyan.
    
    Args:
        text: Message text
        prefix: Prefix before message (default: "[INFO]")
        
    Example:
        >>> print_info("Starting optimization")
        [INFO] Starting optimization
    """
    print_colored(f"{prefix} {text}", Colors.INFO)


def print_debug(text: str, prefix: str = "[DEBUG]"):
    """
    Print debug message in magenta.
    
    Args:
        text: Message text
        prefix: Prefix before message (default: "[DEBUG]")
        
    Example:
        >>> print_debug("Chromosome decoded")
        [DEBUG] Chromosome decoded
    """
    print_colored(f"{prefix} {text}", Colors.DEBUG)


def print_progress(current: int, total: int, text: str = "Progress"):
    """
    Print progress indicator.
    
    Args:
        current: Current step
        total: Total steps
        text: Description text
        
    Example: 
        >>> print_progress(5, 40, "Evaluating generation")
        [INFO] Evaluating generation:  5/40 (12.5%)
    """
    percentage = (current / total) * 100 if total > 0 else 0
    print_info(f"{text}: {current}/{total} ({percentage:.1f}%)")


def print_key_value(key: str, value: str, key_color: str = Colors.INFO, 
                   value_color: str = ""):
    """
    Print key-value pair with different colors.
    
    Args:
        key: Key name
        value: Value
        key_color: Color for key (default: INFO cyan)
        value_color: Color for value (default: no color)
        
    Example: 
        >>> print_key_value("Dataset", "SECOM")
        Dataset: SECOM
    """
    print(f"{colored(key, key_color)}: {colored(str(value), value_color)}")


def print_list(items: list, bullet: str = "  -", color: str = ""):
    """
    Print a list of items with bullets.
    
    Args:
        items: List of items to print
        bullet: Bullet character (default: "  -")
        color: Color for items (default: no color)
        
    Example:
        >>> print_list(["Item 1", "Item 2", "Item 3"], color=Colors.INFO)
          - Item 1
          - Item 2
          - Item 3
    """
    for item in items:
        print_colored(f"{bullet} {item}", color)


def print_table_row(columns: list, widths: list, color: str = ""):
    """
    Print a table row with fixed column widths.
    
    Args:
        columns: List of column values
        widths: List of column widths
        color: Color for row (default: no color)
        
    Example:
        >>> print_table_row(["Name", "Value", "Status"], [20, 15, 10])
        Name                Value          Status
    """
    row = ""
    for col, width in zip(columns, widths):
        row += str(col).ljust(width)
    print_colored(row, color)


def disable_colors():
    """
    Disable all colors (useful for file output or non-color terminals).
    
    After calling this, all color codes become empty strings.
    """
    # Reinitialize colorama with strip=True to remove all color codes
    init(autoreset=True, strip=True)


def enable_colors():
    """
    Re-enable colors after they've been disabled.
    """
    init(autoreset=True, strip=False)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == '__main__':
    print_header("COLOR SYSTEM DEMO")
    
    print("\nSemantic Colors:")
    print_success("This is a success message")
    print_error("This is an error message")
    print_warning("This is a warning message")
    print_info("This is an info message")
    print_debug("This is a debug message")
    
    print("\n" + "-" * 70)
    print_section("Custom Formatting")
    
    print("\nKey-Value Pairs:")
    print_key_value("Dataset", "SECOM", Colors.INFO, Colors.SUCCESS)
    print_key_value("Model", "Neural Network", Colors.INFO, Colors.SUCCESS)
    print_key_value("HPO Method", "NSGA-II", Colors.INFO, Colors.SUCCESS)
    
    print("\nLists:")
    print_colored("Available options:", Colors.INFO, bold=True)
    print_list(["Option 1", "Option 2", "Option 3"], color=Colors.INFO)
    
    print("\nProgress:")
    print_progress(5, 40, "Evaluating generation")
    print_progress(20, 40, "Evaluating generation")
    print_progress(40, 40, "Evaluating generation")
    
    print("\nTable:")
    print_table_row(["Parameter", "Value", "Type"], [25, 20, 15], Colors.HEADER)
    print_table_row(["num_layers", "3", "discrete"], [25, 20, 15])
    print_table_row(["learning_rate", "0.001", "continuous"], [25, 20, 15])
    print_table_row(["activation", "relu", "categorical"], [25, 20, 15])
    
    print("\nRaw Colors:")
    print(colored("Green text", Colors.GREEN))
    print(colored("Red text", Colors.RED))
    print(colored("Yellow text", Colors.YELLOW))
    print(colored("Blue text", Colors.BLUE))
    print(colored("Magenta text", Colors.MAGENTA))
    print(colored("Cyan text", Colors.CYAN))
    
    print("\nBold/Bright:")
    print(colored("Normal text", Colors.INFO))
    print(colored("Bold text", Colors.INFO, bold=True))
    
    print_header("DEMO COMPLETE")