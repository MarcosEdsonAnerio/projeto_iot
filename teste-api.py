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
from jose import jwt  # Para autenticação JWT
import numpy as np  # Garante que NumPy está disponível como 'np'

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configurações principais
BASE_URL = "http://192.168.100.160:3000/ensino/turma/a31551e7-1017-4a53-9235-6d7860083aed/alunos-fotos"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuaWNrbmFtZSI6IkFkbWluaXN0cmFkb3IiLCJyb2xlIjoiYWRtaW4iLCJzdWIiOiJjODc3YjFjYy1mMjNlLTQwY2EtYjhlNy1lY2Y4YjA4YjJkZDYiLCJwbGF0Zm9ybSI6IndlYiIsImlhdCI6MTczNDM5NDQ3MywiZXhwIjoxNzM0Mzk4MDczfQ.py4EYg1C2GJCgCetmc_fW9xE_SelLMXYYILQNH9cwD4"
HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}
IMAGES_DIR = r"./images"
UNKNOWN_DIR = os.path.join(IMAGES_DIR, "unknown")
THRESHOLD = 0.6  # Similaridade mínima
FRAME_SKIP = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                aluno_id = aluno["id"]
                aluno_nome = aluno["pessoa"]["nome"].replace(" ", "_")
                foto_url = aluno["pessoa"].get("foto")

                # Criar pasta do aluno
                aluno_pasta = os.path.join(IMAGES_DIR, f"{aluno_nome}_{aluno_id}")
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
        arquivos = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(('.jpg', '.jpeg', '.png'))]
        arquivos.sort(key=os.path.getmtime, reverse=True)  # Ordena pelo tempo de modificação (mais recente primeiro)

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

# Processamento de cada frame
def process_frame(frame, face_app, reference_embeddings):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        live_embedding = face.embedding

        best_similarity = 0
        best_aluno = "Desconhecido"

        # Validar o embedding da face detectada
        if live_embedding is not None and len(live_embedding) > 0:
            for aluno, ref_embeddings in reference_embeddings.items():
                # Verificar se ref_embeddings não está vazio
                if ref_embeddings and all(isinstance(e, np.ndarray) and len(e) > 0 for e in ref_embeddings):
                    try:
                        ref_embeddings_array = np.vstack(ref_embeddings)  # Garante que é 2D
                        similarity = max(cosine_similarity([live_embedding], ref_embeddings_array)[0])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_aluno = aluno
                    except ValueError as e:
                        logging.error(f"Erro ao calcular similaridade para {aluno}: {e}")
                        continue

        # Adicionar texto e retângulo no frame
        color = (0, 255, 0) if best_similarity >= THRESHOLD else (0, 0, 255)
        label = f"{best_aluno}" if best_similarity >= THRESHOLD else "Desconhecido"
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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