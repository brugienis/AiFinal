"""
Train model.
"""

import time
from datetime import datetime


def get_formatted_time():
    """
    Return formatted time.
    :return:
    """
    now = datetime.now()  # current date and time
    return now.strftime("%H:%M:%S")
