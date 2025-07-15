import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
"""Below Line Explanation:
Creating a directory named `log`, in the `Current Working Directory` (using os.getcwd()), and then adding Log file with names as:
`logs_dd_mm_yyyy_HH_MM_SS.log`
"""
LOGS_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_PATH, exist_ok=True)  # Even if the directory already exists, keep appending the log files.

LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
