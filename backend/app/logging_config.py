import logging.config
import os

from app.config import Settings

def setup_logging(settings: Settings):
    """
    logging.conf 파일을 읽어 로깅 설정을 구성
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    logging.config.fileConfig(settings.LOGGER_CONFIG_PATH, disable_existing_loggers=False)

