from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class AppConfig:
    """Application configuration and constants."""
    
    # Appearance
    APPEARANCE_MODE: str = "dark"
    DEFAULT_COLOR_THEME: str = "blue"
    
    # Window
    WINDOW_TITLE: str = "AURAK Shuttle Arrival Predictor"
    WINDOW_SIZE: str = "1200x800"
    MIN_WINDOW_SIZE: Tuple[int, int] = (800, 600)
    
    # Colors
    PALETTE: Dict[str, Tuple[str, str]] = None
    
    # Logging
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        import os
        import sys
        
        # Determine base path
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            # When running from source, we are now in src/config.py, so we need to go up one level
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.BASE_PATH = base_path
        self.ASSETS_DIR = os.path.join(base_path, "assets")
        self.LOG_FILE = os.path.join(base_path, "app.log")

        if self.PALETTE is None:
            self.PALETTE = {
                "panel_fg": ("#f8fafc", "#0f172a"),  # Main panel glass
                "panel_border": ("#e2e8f0", "#1e293b"),  # Subtle border
                "section_fg": ("#ffffff", "#1f293b"),  # Inner sections
                "text_primary": ("#0f172a", "#f8fafc"),
                "text_secondary": ("#64748b", "#94a3b8"),
                "accent_primary": ("#2563eb", "#38bdf8"),
                "accent_secondary": ("#c2410c", "#f97316"),
                "status_success": ("#047857", "#10b981"),
                "status_warning": ("#f97316", "#f97316"),
                "status_danger": ("#c2410c", "#ef4444"),
            }

    # Typography
    FONTS: Dict[str, Dict] = None
    
    def get_fonts(self, ctk_font_class):
        """Return font configurations using the provided CTkFont class."""
        return {
            "h1": ctk_font_class(size=28, weight="bold"),
            "h2": ctk_font_class(size=22, weight="bold"),
            "h3": ctk_font_class(size=16, weight="bold"),
            "h4": ctk_font_class(size=24, weight="bold"),
            "body": ctk_font_class(size=14),
            "body_small": ctk_font_class(size=12),
            "body_bold": ctk_font_class(size=13, weight="bold"),
            "code": ctk_font_class(family="Courier", size=10),
        }

    # Gradient Palettes
    GRADIENT_PALETTES = [
        ("#0f172a", "#1e293b"),
        ("#111827", "#1e3a8a"),
        ("#0b1120", "#164e63"),
        ("#1f2937", "#0f172a"),
    ]
    
    # Simulation
    SIMULATION_SEGMENTS: int = 320
    ANIMATION_SPEED_ON_TIME: float = 6.0
    ANIMATION_SPEED_LATE: float = 3.0
    
    # Logging
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
