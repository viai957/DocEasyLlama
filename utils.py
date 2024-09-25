import logging

# Configure the logger
def setup_logger(log_file: str = 'application.log'):
    """Sets up a logger that writes both to console and a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Formatter for logging output
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Adding the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logger()

def log_info(message: str):
    """Log informational messages."""
    logger.info(message)

def log_error(message: str):
    """Log error messages."""
    logger.error(message)

def log_warning(message: str):
    """Log warning messages."""
    logger.warning(message)

def log_debug(message: str):
    """Log debug messages."""
    logger.debug(message)

# Example helper function
def is_valid_query(query: str) -> bool:
    """Check if a query string is valid (non-empty and non-whitespace)."""
    if query.strip():
        return True
    log_warning("Received an invalid query: empty or whitespace.")
    return False
