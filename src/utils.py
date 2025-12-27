import logging
import platform
import subprocess
import shutil
import threading
import sys
from .config import AppConfig

class DataLoadError(Exception):
    """Custom exception raised when data loading fails."""
    pass

def setup_logging():
    """Configure logging to file and console."""
    config = AppConfig()
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(config.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging initialized.")

def play_sound_cross_platform(path: str) -> bool:
    """
    Attempt to play a sound non‑blocking on Windows/macOS/Linux.
    Handles missing drivers or files gracefully.
    Returns True if a method was started successfully, else False.
    """
    try:
        if platform.system() == 'Windows':
            try:
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return True
            except ImportError:
                logging.warning("winsound module not available.")
            except Exception as e:
                logging.error(f"Failed to play sound on Windows: {e}")
                
        # macOS uses 'afplay'
        elif platform.system() == 'Darwin':
            if shutil.which('afplay'):
                subprocess.Popen(['afplay', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
                
        # Linux: try a few common players
        else:
            for cmd in ('aplay', 'paplay', 'ffplay', 'play'):
                if shutil.which(cmd):
                    subprocess.Popen([cmd, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                    
    except Exception as e:
        logging.error(f"Unexpected error playing sound: {e}")
        
    return False
