import requests
import os
import logging
import time
from datetime import datetime
from .utils import baixar_foto
from .config import BASE_URL, IMAGES_DIR, TURMA_ID


def fetch_aluno_photos():
    JWT_TOKEN = os.getenv("JWT_TOKEN")
    HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}

    logging.info("Buscando novas fotos dos alunos...")
    try:
        response = requests.get(
            f"{BASE_URL}/ensino/turma/{TURMA_ID}/alunos-fotos", headers=HEADERS
        )

        if response.status_code == 200:
            alunos = response.json()
            for aluno in alunos:
                matricula_id = aluno["matricula_id"]
                aluno_nome = aluno["pessoa"]["nome"].replace(" ", "_")
                foto_url = aluno["pessoa"].get("foto")

                aluno_pasta = os.path.join(IMAGES_DIR, f"{matricula_id}")
                os.makedirs(aluno_pasta, exist_ok=True)

                if foto_url:
                    foto_path = os.path.join(aluno_pasta, os.path.basename(foto_url))
                    if not os.path.exists(foto_path):
                        baixar_foto(foto_url, aluno_pasta)
                    else:
                        logging.info(f"Foto já existe para o aluno {aluno_nome}")
        else:
            logging.error(f"Erro ao buscar fotos: {response.status_code}")
    except Exception as e:
        logging.error(f"Erro ao buscar fotos: {e}")


def schedule_fetch_photos():
    while True:
        fetch_aluno_photos()
        logging.info("Aguardando 12 horas para nova busca...")
        time.sleep(12 * 3600)
