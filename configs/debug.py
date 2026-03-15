# configs/debug.py

import os
from datetime import datetime

# Toggle debug mode here or via environment variable
DEBUG = os.getenv("OMNIRAG_DEBUG", "true").lower() == "true"

def log(message: str, component: str = "SYSTEM"):
    """
    Standard debug logger for OmniRAG.

    Usage:
        log("Something happened", "Runner")
    """
    if not DEBUG:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{component}] {message}")
