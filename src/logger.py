import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class Logger:
    def __init__(self, name: str = 'ai_tutor', log_dir: str = 'logs'):
        '''Initialize the logger with a name and log directory'''
        self.name = name
        self.log_dir = log_dir
        
        # Create the log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate the log file name
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.log_file = os.path.join(log_dir, f'{name}_{current_date}.log')

        # Initialize the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers to avoid duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create file handler for both file and console
        self._setup_file_handler()
        self._setup_console_handler()

        # Set formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to handlers
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def _setup_file_handler(self):
        '''Setup the file handler'''
        # Rotating file handler with max 5MB and 3 backups
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=5*1024*1024,
            backupCount=3
        )

        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        '''Setup the console handler'''
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        '''Return the configured logger instance'''
        return self.logger

def get_logger(name: str = 'ai_tutor', log_dir: str = 'logs'):
    '''Get or create a logger instance'''
    return Logger(name, log_dir).get_logger()
