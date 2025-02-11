import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
AULA_ID = os.getenv("AULA_ID")
TURMA_ID = os.getenv("TURMA_ID")
IMAGES_DIR = os.getenv("IMAGES_DIR")
UNKNOWN_DIR = os.getenv("UNKNOWN_DIR")
THRESHOLD = float(os.getenv("THRESHOLD"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP"))
JWT_TOKEN = None
