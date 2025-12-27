import customtkinter as ctk
from src.gui import AURAKShuttlePredictor
from src.utils import setup_logging

if __name__ == "__main__":
    # Initialize logging
    setup_logging()
    
    # Create and run application
    app = AURAKShuttlePredictor()
    app.mainloop()
