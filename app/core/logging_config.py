import logging
import sys

# Log Formatı: ZAMAN - SEVİYE - MESAJ
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"

def setup_logger(name: str = "ML_App"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Eğer daha önce handler eklenmediyse ekle (Çift log yazmayı önler)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Global logger nesnesi
logger = setup_logger()