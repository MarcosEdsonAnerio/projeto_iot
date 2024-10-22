import os
import cv2
import torch
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from shutil import move, rmtree

# Configuração de logging para monitorar operações
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios
IMAGES_DIR = r"C:\Users\helya\OneDrive\Área de Trabalho\Marcos\Projeto iot\insightface-master\images"
UNKNOWN_DIR = os.path.join(IMAGES_DIR, "unknown")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.5  # 50% de similaridade para excluir

# Configurações iniciais
def initialize_directories():
    if not os.path.exists(UNKNOWN_DIR):
        os.makedirs(UNKNOWN_DIR)
    logging.info(f"Diretórios verificados: {IMAGES_DIR}, {UNKNOWN_DIR}")

# Inicializar o modelo de reconhecimento facial InsightFace
def initialize_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Inicializar o modelo YOLOv5 para detecção de roupas e acessórios
def initialize_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Função para calcular a similaridade cosseno
def cos_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Função para carregar embeddings das imagens dos alunos na pasta "images"
def load_reference_embeddings(directory, face_app):
    embeddings = {}
    
    # Percorrer as subpastas da pasta 'images' onde cada subpasta é o nome do aluno
    for aluno_folder in os.listdir(directory):
        aluno_folder_path = os.path.join(directory, aluno_folder)
        
        if os.path.isdir(aluno_folder_path):  # Verificar se é uma subpasta
            embeddings[aluno_folder] = []  # Inicializar a lista de embeddings para o aluno
            
            # Carregar todas as imagens da subpasta do aluno
            for img_file in os.listdir(aluno_folder_path):
                img_path = os.path.join(aluno_folder_path, img_file)
                
                if img_file.lower().endswith(IMAGE_EXTENSIONS):
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = face_app.get(image_rgb)
                    if len(faces) == 0:
                        continue
                    
                    # Obter o embedding da primeira face detectada
                    face_embedding = faces[0].embedding
                    embeddings[aluno_folder].append(face_embedding)  # Adicionar embedding ao aluno
                    
    logging.info("Embeddings carregados.")
    return embeddings

# Função para salvar a imagem em uma pasta (unknown ou aluno correspondente)
def save_image_in_folder(frame, folder_name, aluno_name=None):
    folder_path = os.path.join(folder_name, aluno_name) if aluno_name else folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"captured_{timestamp}.jpg"
    image_path = os.path.join(folder_path, image_filename)
    cv2.imwrite(image_path, frame)
    logging.info(f"Imagem salva em {folder_path} como {image_filename}")

# Função para identificar múltiplos rostos e evitar exclusões incorretas
def identify_and_check_duplicates(frame, face_app, yolo_model, reference_embeddings, threshold=THRESHOLD):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_frame)

    current_faces = []  # Armazena os rostos processados

    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        live_embedding = face.embedding
        current_faces.append((face, live_embedding))  # Adiciona o rosto e o embedding à lista

        # Analisar com embeddings de referência
        best_similarity = 0
        best_aluno = None

        # Verificar todos os alunos registrados para encontrar a melhor correspondência
        for aluno, ref_embeddings in reference_embeddings.items():
            for ref_embedding in ref_embeddings:
                similarity = cos_sim(ref_embedding, live_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_aluno = aluno

        # Verificar se o aluno já é conhecido
        if best_similarity >= threshold:
            label = f"Identificado: {best_aluno} - Similaridade: {best_similarity:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Atualizar os embeddings para lidar com mudanças no rosto
            save_image_in_folder(frame, IMAGES_DIR, best_aluno)
        else:
            label = f"Desconhecido - Similaridade: {best_similarity:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Salvar na pasta unknown
            save_image_in_folder(frame, UNKNOWN_DIR)

# Função para normalizar iluminação da imagem
def normalize_lighting(image):
    """Normaliza a iluminação da imagem usando equalização de histograma."""
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_normalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return image_normalized

# Inicialização do sistema
def main():
    initialize_directories()
    face_app = initialize_face_model()
    yolo_model = initialize_yolo_model()

    # Carregar embeddings existentes das subpastas em "images"
    reference_embeddings = load_reference_embeddings(IMAGES_DIR, face_app)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Normalizar iluminação da imagem
        frame = normalize_lighting(frame)

        # Identificar e verificar duplicatas
        identify_and_check_duplicates(frame, face_app, yolo_model, reference_embeddings)

        cv2.imshow("Verificação Multimodal de Indivíduos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
