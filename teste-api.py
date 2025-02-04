import os
import cv2
import torch
import requests
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from shutil import move
from jose import jwt
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configurações principais
BASE_URL = "http://192.168.61.160:3000/ensino/turma/4614f88d-701f-43f3-b099-d0562a5cae6e/alunos-fotos"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuaWNrbmFtZSI6IkFkbWluaXN0cmFkb3IiLCJyb2xlIjoiYWRtaW4iLCJzdWIiOiJjODc3YjFjYy1mMjNlLTQwY2EtYjhlNy1lY2Y4YjA4YjJkZDYiLCJwbGF0Zm9ybSI6IndlYiIsImlhdCI6MTczODYyOTI1MiwiZXhwIjoxNzM4NjMyODUyfQ.u5HI8hvJvVNAJNTGU6b_Y4l-4cXk62L_MTkiarxvA8Y"
HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}
IMAGES_DIR = r"./images"
UNKNOWN_DIR = os.path.join(IMAGES_DIR, "unknown")
THRESHOLD = 0.6
FRAME_SKIP = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Inicialização de câmera
def initialize_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    cap.set(cv2.CAP_PROP_FPS, 120)
    return cap

    # ip_webcam_url = "http://192.168.61.242:8889/video"
    # cap = cv2.VideoCapture(ip_webcam_url)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    # return cap


# Inicialização de diretórios
def initialize_directories():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    logging.info(f"Diretórios verificados: {IMAGES_DIR}, {UNKNOWN_DIR}")


# Inicializar modelos
def initialize_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if DEVICE.type == "cuda" else -1, det_size=(640, 640))
    return app


def initialize_yolo_model():
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
    return model


# Função para buscar fotos
def fetch_aluno_photos():
    logging.info("Buscando novas fotos dos alunos...")
    try:
        response = requests.get(BASE_URL, headers=HEADERS)
        if response.status_code == 200:
            alunos = response.json()
            for aluno in alunos:
                matricula_id = aluno["matricula_id"]
                aluno_nome = aluno["pessoa"]["nome"].replace(" ", "_")
                foto_url = aluno["pessoa"].get("foto")

                # Criar pasta do aluno
                aluno_pasta = os.path.join(IMAGES_DIR, f"{matricula_id}")
                os.makedirs(aluno_pasta, exist_ok=True)

                # Baixar foto se existir
                if foto_url:
                    baixar_foto(foto_url, aluno_pasta)
        else:
            logging.error(f"Erro ao buscar fotos: {response.status_code}")
    except Exception as e:
        logging.error(f"Erro ao buscar fotos: {e}")


# Função para baixar uma foto
def baixar_foto(url, pasta):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Criar nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(pasta, f"{timestamp}.jpg")

            # Salvar a nova foto
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Foto salva: {file_path}")

            # Manter apenas as 5 fotos mais recentes
            manter_ultimas_fotos(pasta, limite=5)

    except Exception as e:
        logging.error(f"Erro ao baixar foto {url}: {e}")


# Função auxiliar para manter apenas as últimas 'limite' fotos na pasta
def manter_ultimas_fotos(pasta, limite=5):
    try:
        # Listar todas as fotos da pasta
        arquivos = [
            os.path.join(pasta, f)
            for f in os.listdir(pasta)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        arquivos.sort(
            key=os.path.getmtime, reverse=True
        )  # Ordena pelo tempo de modificação (mais recente primeiro)

        # Deletar arquivos além do limite
        if len(arquivos) > limite:
            for arquivo in arquivos[limite:]:
                os.remove(arquivo)
                logging.info(f"Arquivo removido: {arquivo}")

    except Exception as e:
        logging.error(f"Erro ao manter últimas fotos na pasta {pasta}: {e}")


# Carregar embeddings das imagens dos alunos
def load_reference_embeddings(directory, face_app):
    embeddings = {}
    for aluno_folder in os.listdir(directory):
        aluno_pasta = os.path.join(directory, aluno_folder)
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


# Função para enviar requisição HTTP quando um aluno for identificado
def send_identification_request(aluno_nome):
    try:
        response = requests.post(BASE_URL, headers=HEADERS, json={"aluno": aluno_nome})
        if response.status_code == 200:
            logging.info(f"Requisição enviada com sucesso para {aluno_nome}")
        else:
            logging.error(
                f"Erro ao enviar requisição para {aluno_nome}: {response.status_code}"
            )
    except Exception as e:
        logging.error(f"Erro ao enviar requisição para {aluno_nome}: {e}")


# Processamento de cada frame
def process_frame(frame, face_app, reference_embeddings):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        live_embedding = face.embedding

        best_similarity = 0
        best_aluno = "Desconhecido"
        best_matricula = None

        # Validar o embedding da face detectada
        if live_embedding is not None and len(live_embedding) > 0:
            for aluno, ref_embeddings in reference_embeddings.items():
                # Verificar se ref_embeddings não está vazio
                if ref_embeddings and all(
                    isinstance(e, np.ndarray) and len(e) > 0 for e in ref_embeddings
                ):
                    try:
                        ref_embeddings_array = np.vstack(
                            ref_embeddings
                        )  # Garante que é 2D
                        similarity = max(
                            cosine_similarity([live_embedding], ref_embeddings_array)[0]
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_aluno = aluno
                            # Extrair matrícula do nome da pasta
                            best_matricula = aluno.split("_")[-1]
                    except ValueError as e:
                        logging.error(
                            f"Erro ao calcular similaridade para {aluno}: {e}"
                        )
                        continue

        # Adicionar texto e retângulo no frame
        color = (0, 255, 0) if best_similarity >= THRESHOLD else (0, 0, 255)
        label = f"{best_aluno}" if best_similarity >= THRESHOLD else "Desconhecido"
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(
            frame,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        # Enviar requisição HTTP se um aluno for identificado
        if best_similarity >= THRESHOLD and best_matricula:
            url = "http://192.168.61.160:3000/ensino/diario/aulas/frequencia"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuaWNrbmFtZSI6IkFkbWluaXN0cmFkb3IiLCJyb2xlIjoiYWRtaW4iLCJzdWIiOiJjODc3YjFjYy1mMjNlLTQwY2EtYjhlNy1lY2Y4YjA4YjJkZDYiLCJwbGF0Zm9ybSI6IndlYiIsImlhdCI6MTczODYyOTI1MiwiZXhwIjoxNzM4NjMyODUyfQ.u5HI8hvJvVNAJNTGU6b_Y4l-4cXk62L_MTkiarxvA8Y",
            }
            data = {
                "aulas": [
                    {
                        "aula": {"id": "0446c265-7c8b-43e7-95df-0ce1023fad77"},
                        "matricula": {"id": best_matricula},
                        "frequencia": "P",
                    }
                ]
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                logging.info(f"Requisição enviada com sucesso para {best_aluno}")
            else:
                logging.error(
                    f"Erro ao enviar requisição para {best_aluno}: {response.status_code}"
                )

    return frame


# Função agendada para buscar fotos a cada 12 horas
def schedule_fetch_photos():
    while True:
        fetch_aluno_photos()
        logging.info("Aguardando 12 horas para nova busca...")
        time.sleep(12 * 3600)


# Main
def main():
    initialize_directories()
    face_app = initialize_face_model()
    yolo_model = initialize_yolo_model()

    threading.Thread(target=schedule_fetch_photos, daemon=True).start()

    cap = initialize_camera()

    executor = ThreadPoolExecutor(max_workers=2)
    frame_count = 0
    reference_embeddings = load_reference_embeddings(IMAGES_DIR, face_app)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        future = executor.submit(process_frame, frame, face_app, reference_embeddings)
        processed_frame = future.result()

        cv2.imshow("Verificação Multimodal de Indivíduos", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
