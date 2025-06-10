import os
import logging
from datetime import datetime

def setup_logger():
    """Configure and return a logger instance"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logger = logging.getLogger('CrimeAI')
    logger.setLevel(logging.DEBUG)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/crime_ai_{current_time}.log', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 