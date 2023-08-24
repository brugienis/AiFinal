"""
Train model.
"""

import time
from datetime import datetime


def get_formatted_time():
    now = datetime.now()  # current date and time
    return now.strftime("%H:%M:%S")
