import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

DEFAULT_IMAGE_SIZE = 320
DEFAULT_NUM_CLUSTERS = 64
DEFAULT_FEATURE_DIM = 2048
DEFAULT_MAX_DISTANCE = 0.25
DEFAULT_VIDEO_PATH = 'vpr_data/test.MOV'

DEFAULT_MODEL_PATH = 'models/netvlad.pth'
DEFAULT_INDEX_PATH = 'index/faiss_index.bin'

DEFAULT_REDIS_HOST = 'localhost'
DEFAULT_REDIS_PORT = 6379

def str_to_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def str_to_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

IMAGE_SIZE = str_to_int(os.getenv('IMAGE_SIZE'), DEFAULT_IMAGE_SIZE)
NUM_CLUSTERS = str_to_int(os.getenv('NUM_CLUSTERS'), DEFAULT_NUM_CLUSTERS)
FEATURE_DIM = str_to_int(os.getenv('FEATURE_DIM'), DEFAULT_FEATURE_DIM)
MAX_DISTANCE = str_to_float(os.getenv('MAX_DISTANCE'), DEFAULT_MAX_DISTANCE)
VIDEO_PATH = os.getenv('VIDEO_PATH', DEFAULT_VIDEO_PATH)

MODEL_PATH = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
INDEX_PATH = os.getenv('INDEX_PATH', DEFAULT_INDEX_PATH)

REDIS_HOST = os.getenv('REDIS_HOST', DEFAULT_REDIS_HOST)
REDIS_PORT = str_to_int(os.getenv('REDIS_PORT'), DEFAULT_REDIS_PORT)

CONFIG = {
    'image_size': IMAGE_SIZE,
    'num_clusters': NUM_CLUSTERS,
    'feature_dim': FEATURE_DIM,
    'max_distance': MAX_DISTANCE,
    'video_path': VIDEO_PATH,
    'model_path': MODEL_PATH,
    'index_path': INDEX_PATH,
    'redis_host': REDIS_HOST,
    'redis_port': REDIS_PORT,
}
