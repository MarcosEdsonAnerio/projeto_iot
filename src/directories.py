import os
import logging

IMAGES_DIR = r"./images"
UNKNOWN_DIR = os.path.join(IMAGES_DIR, "unknown")


def initialize_directories():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    logging.info(f"Diretórios verificados: {IMAGES_DIR}, {UNKNOWN_DIR}")
