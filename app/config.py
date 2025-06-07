import os
from dotenv import load_dotenv
from pathlib import Path
from .utils import str_to_float, str_to_int

# --- Пути и загрузка переменных окружения ---
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(ENV_PATH)

# --- Значения по умолчанию ---
DEFAULTS = {
    'IMAGE_SIZE': 320,
    'TEST_VIDEO_PATH': 'vpr_data/IMG_0798.MOV',
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': 6379,
}

# --- Конфигурация с загрузкой из окружения и fallback на DEFAULTS ---
IMAGE_SIZE = str_to_int(os.getenv('IMAGE_SIZE'), DEFAULTS['IMAGE_SIZE'])
TEST_VIDEO_PATH = os.getenv('TEST_VIDEO_PATH', DEFAULTS['TEST_VIDEO_PATH'])
REDIS_HOST = os.getenv('REDIS_HOST', DEFAULTS['REDIS_HOST'])
REDIS_PORT = str_to_int(os.getenv('REDIS_PORT'), DEFAULTS['REDIS_PORT'])

# --- Словарь конфигурации для удобного доступа ---
CONFIG = {
    'image_size': IMAGE_SIZE,
    'test_video_path': TEST_VIDEO_PATH,
    'redis': {
        'host': REDIS_HOST,
        'port': REDIS_PORT,
    }
}