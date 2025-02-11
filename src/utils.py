import os
import logging
import requests
from datetime import datetime
import cv2
import numpy as np
from .config import IMAGES_DIR
import re


def baixar_foto(url, pasta):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            match = re.search(r"/([a-f0-9\-]+)\.jpeg", url)
            if match:
                file_name = match.group(1) + ".jpeg"
            else:
                raise ValueError("UUID não encontrado no URL")

            file_path = os.path.join(pasta, file_name)

            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Foto salva: {file_path}")

            manter_ultimas_fotos(pasta, limite=5)
    except Exception as e:
        logging.error(f"Erro ao baixar foto {url}: {e}")


def manter_ultimas_fotos(pasta, limite=4):
    try:
        arquivos = [
            os.path.join(pasta, f)
            for f in os.listdir(pasta)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        arquivos.sort(key=os.path.getmtime, reverse=True)

        if len(arquivos) > limite:
            print(arquivos)
            for arquivo in arquivos[limite:]:
                os.remove(arquivo)
                logging.info(f"Arquivo removido: {arquivo}")
    except Exception as e:
        logging.error(f"Erro ao manter últimas fotos na pasta {pasta}: {e}")


def load_reference_embeddings(face_app):
    embeddings = {}
    for aluno_folder in os.listdir(IMAGES_DIR):
        aluno_pasta = os.path.join(IMAGES_DIR, aluno_folder)
        if os.path.isdir(aluno_pasta):
            embeddings[aluno_folder] = []
            for img_file in os.listdir(aluno_pasta):
                img_path = os.path.join(aluno_pasta, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = face_app.get(image_rgb)
                if faces:
                    embeddings[aluno_folder].append(faces[0].embedding)
    logging.info("Embeddings carregados.")
    return embeddings
