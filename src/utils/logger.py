import logging
from threading import Lock

# ANSI color codes for terminal output
class LogColors:
    """
    This class contains ANSI color codes used for formatting log messages 
    with colors in the terminal.
    """
    INFO_BOLD = "\033[1;32m"  # Bright green (bold) for INFO
    INFO = "\033[0;32m"  # Soft green for INFO
    WARNING_BOLD = "\033[1;33m"  # Bright yellow (bold) for WARNING
    WARNING = "\033[0;33m"  # Soft yellow for WARNING
    ERROR_BOLD = "\033[1;31m"  # Bright red (bold) for ERROR
    ERROR = "\033[0;31m"  # Soft red for ERROR
    RESET = "\033[0m"  # Reset color formatting


class ColoredFormatter(logging.Formatter):
    """
    A custom log formatter that adds color to the log messages based on their
    severity (INFO, WARNING, ERROR).
    """
    def format(self, record):
        """
        Override the format method to apply color formatting to log levels
        and messages based on their severity.

        Parameters:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with color.
        """
        # Check if the message is empty and return a blank line without other info
        if not record.msg.strip():  # If the message is empty or contains only spaces
            return ""

        # Apply color formatting to log levels and messages
        if record.levelname == "INFO":
            record.levelname = f"{LogColors.INFO_BOLD}{record.levelname}{LogColors.RESET}"
            record.msg = f"{LogColors.INFO}{record.msg}{LogColors.RESET}"
        elif record.levelname == "WARNING":
            record.levelname = f"{LogColors.WARNING_BOLD}{record.levelname}{LogColors.RESET}"
            record.msg = f"{LogColors.WARNING}{record.msg}{LogColors.RESET}"
        elif record.levelname == "ERROR":
            record.levelname = f"{LogColors.ERROR_BOLD}{record.levelname}{LogColors.RESET}"
            record.msg = f"{LogColors.ERROR}{record.msg}{LogColors.RESET}"

        return super().format(record)


class LoggerSingleton:
    """
    Singleton Logger class to ensure that only one logger instance is used
    throughout the application. It also handles thread safety.
    """
    _instance = None  # Holds the single instance of LoggerSingleton
    _lock = Lock()  # Ensures thread safety when creating the instance
    
    def __new__(cls):
        """
        Create or return the singleton instance of the LoggerSingleton class.

        Returns:
            LoggerSingleton: The singleton logger instance.
        """
        # Locking mechanism to prevent race conditions in multithreaded environments
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_logger()
        return cls._instance
    
    def _init_logger(self):
        """
        Initialize the logger with a console handler and a custom formatter
        that applies color formatting based on the log level.

        This method sets up a named logger with the level set to INFO and
        removes any existing handlers to avoid duplicate logs.
        """
        # Initialize the logger
        self.logger = logging.getLogger("MiLogger")  # Create a named logger
        self.logger.setLevel(logging.INFO)  # Set the logging level to INFO
        self.logger.handlers.clear()  # Remove existing handlers (if any)
        
        # Create a console handler with color formatting
        console_handler = logging.StreamHandler()
        formatter = ColoredFormatter("%(asctime)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)  # Attach handler to logger
        
        # Clear default root logger handlers to avoid duplicate logs
        # logging.getLogger().handlers.clear()  # You can remove this line if it's not required.

    def get_logger(self):
        """
        Returns the singleton logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        # Log an initial message for testing purposes
        self.logger.info("")  # Log a blank line without timestamp or other info
        self.logger.info("Initializing logger...")
        self.logger.info("=========================================")
        self.logger.info("Logger initialized successfully.")
        self.logger.warning("This is a warning message.")
        self.logger.error("This is an error message.")
        self.logger.info("=========================================")
        self.logger.info("")  # Log another blank line without timestamp or other info
        
        return self.logger
