import os
import cv2
import torch
import numpy as np
import logging
import threading
import zipfile
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from deepface import DeepFace
from shutil import move, rmtree
import smtplib

# Configuração de logging para monitorar operações
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios de armazenagem
ALUNOS_DIR = "alunos"
CAPTURED_IMAGES_DIR = "captured_images"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.5  # 50% de similaridade para excluir
FRAME_INTERVAL = 5  # Processar a cada 5 frames

# Configurações iniciais
def initialize_directories():
    if not os.path.exists(ALUNOS_DIR):
        os.makedirs(ALUNOS_DIR)
    if not os.path.exists(CAPTURED_IMAGES_DIR):
        os.makedirs(CAPTURED_IMAGES_DIR)
    logging.info(f"Diretórios verificados: {ALUNOS_DIR}, {CAPTURED_IMAGES_DIR}")

# Inicializar o modelo de reconhecimento facial InsightFace
def initialize_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Inicializar o modelo YOLOv5 para detecção de roupas e acessórios
def initialize_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cuda')  # Forçar uso de GPU

# Função para calcular similaridade cosseno
def cos_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Função para carregar embeddings de imagens existentes
def load_reference_embeddings(directory, face_app):
    embeddings = {}
    for aluno_folder in os.listdir(directory):
        aluno_folder_path = os.path.join(directory, aluno_folder)
        if not os.path.isdir(aluno_folder_path):
            continue
        for img_file in os.listdir(aluno_folder_path):
            img_path = os.path.join(aluno_folder_path, img_file)
            if not img_file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_app.get(image_rgb)
            if len(faces) == 0:
                continue
            face_embedding = faces[0].embedding
            if aluno_folder not in embeddings:
                embeddings[aluno_folder] = []
            embeddings[aluno_folder].append(face_embedding)
    logging.info("Embeddings carregados.")
    return embeddings

# Função para capturar e salvar uma imagem
def save_image_in_new_folder(frame, aluno_num):
    new_folder = f"aluno_{aluno_num}"
    new_folder_path = os.path.join(ALUNOS_DIR, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"captured_{timestamp}.jpg"
    image_path = os.path.join(new_folder_path, image_filename)
    cv2.imwrite(image_path, frame)
    logging.info(f"Nova pasta {new_folder} criada e imagem salva como {image_filename}")

# Função para verificar e renomear o arquivo se já existir no destino
def safe_move_file(source, destination):
    base, ext = os.path.splitext(destination)
    counter = 1
    while os.path.exists(destination):
        destination = f"{base}_{counter}{ext}"
        counter += 1
    move(source, destination)

# Função para reduzir a resolução de um frame para acelerar o processamento
def reduce_resolution(frame, scale_percent=50):
    """Reduz a resolução da imagem para acelerar o processamento."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

# Função para processar YOLOv5 em uma thread separada
def process_yolo(yolo_model, frame):
    return detect_accessories(yolo_model, frame)

# Função para detectar e verificar duplicatas de forma paralela
def identify_and_check_duplicates_parallel(frame, face_app, yolo_model, reference_embeddings, threshold=THRESHOLD):
    # Rodar YOLOv5 em paralelo
    yolo_thread = threading.Thread(target=process_yolo, args=(yolo_model, frame))
    yolo_thread.start()

    # Processar a identificação facial enquanto o YOLOv5 roda
    identify_and_check_duplicates(frame, face_app, yolo_model, reference_embeddings, threshold)

    # Aguardar conclusão do YOLOv5
    yolo_thread.join()

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
            # Verificar duplicatas e excluir apenas se for necessário
            merge_and_remove_duplicates(reference_embeddings, best_aluno, live_embedding)
            # Atualizar embeddings para lidar com mudanças no rosto (ex: cor de cabelo, barba)
            update_embeddings_over_time(reference_embeddings, best_aluno, live_embedding)
        else:
            # Caso seja um novo aluno, verificar se é realmente novo
            if not is_existing_student(live_embedding, reference_embeddings):
                aluno_counter = len(os.listdir(ALUNOS_DIR)) + 1
                save_image_in_new_folder(frame, aluno_counter)
                reference_embeddings[f"aluno_{aluno_counter}"] = [live_embedding]

    # Detectar grupos de pessoas
    detect_groups(reference_embeddings, current_faces)

# Função para verificar se um aluno já está presente no sistema
def is_existing_student(new_embedding, reference_embeddings, threshold=SIMILARITY_THRESHOLD):
    for aluno, embeddings_list in reference_embeddings.items():
        for existing_embedding in embeddings_list:
            similarity = cos_sim(existing_embedding, new_embedding)
            if similarity >= threshold:
                return True
    return False

# Função para atualizar os embeddings de um aluno ao longo do tempo
def update_embeddings_over_time(reference_embeddings, aluno, new_embedding, threshold=0.8):
    """Atualiza os embeddings ao longo do tempo, se houver uma variação significativa."""
    if aluno in reference_embeddings:
        for old_embedding in reference_embeddings[aluno]:
            similarity = cos_sim(old_embedding, new_embedding)
            if similarity < threshold:  # Significativa variação no rosto
                reference_embeddings[aluno].append(new_embedding)
                logging.info(f"Embedding do aluno {aluno} atualizado devido a mudanças físicas.")
                return

# Função para mesclar e remover alunos duplicados em tempo real
def merge_and_remove_duplicates(reference_embeddings, current_aluno, live_embedding, threshold=SIMILARITY_THRESHOLD):
    """Verifica se dois alunos possuem embeddings similares e mescla pastas."""
    for aluno, embeddings_list in reference_embeddings.items():
        if aluno != current_aluno:
            for existing_embedding in embeddings_list:
                similarity = cos_sim(existing_embedding, live_embedding)
                if similarity >= threshold:
                    logging.info(f"Mesclando e removendo {current_aluno} por similaridade de {similarity:.2f} com {aluno}")
                    source_folder = os.path.join(ALUNOS_DIR, current_aluno)
                    dest_folder = os.path.join(ALUNOS_DIR, aluno)

                    # Mover arquivos com verificação de nome
                    for img in os.listdir(source_folder):
                        source_file = os.path.join(source_folder, img)
                        dest_file = os.path.join(dest_folder, img)
                        safe_move_file(source_file, dest_file)

                    # Remover a pasta duplicada
                    rmtree(source_folder)
                    logging.info(f"Pasta duplicada {current_aluno} removida.")
                    del reference_embeddings[current_aluno]
                    return  # Já mesclado, pode parar a verificação para esse aluno

# Função para normalizar iluminação da imagem
def normalize_lighting(image):
    """Normaliza a iluminação da imagem usando equalização de histograma."""
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_normalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return image_normalized

# Função para detectar acessórios como óculos e chapéus
def detect_accessories(yolo_model, frame):
    """Detecta acessórios como óculos e chapéus usando YOLOv5."""
    results = yolo_model(frame)
    accessories = []
    for result in results.pred[0]:
        if result[-1] in [26, 27]:  # Classes de óculos, chapéus (dependendo do modelo YOLO)
            accessories.append(result)
    return accessories

# Função para detectar máscara facial
def detect_mask(yolo_model, frame):
    """Usa YOLO para detectar se o indivíduo está usando uma máscara."""
    results = yolo_model(frame)
    for result in results.pred[0]:
        if result[-1] == 28:  # Classe para máscara facial (dependendo do modelo YOLO)
            return True
    return False

# Função para compactar imagens antigas para economizar espaço
def compress_old_images(aluno_folder):
    """Compacta imagens antigas de um aluno para economizar espaço."""
    zipf = zipfile.ZipFile(f"{aluno_folder}.zip", 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(aluno_folder):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)
    zipf.close()

# Função para enviar um alerta de segurança por email
def send_security_alert(email, message):
    """Envia um alerta de segurança via email."""
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("your_email", "your_password")
    server.sendmail("your_email", email, message)
    server.quit()

# Função para detecção de grupos de pessoas
def detect_groups(reference_embeddings, current_faces):
    """Detecta padrões de grupos de pessoas que aparecem juntas frequentemente."""
    group = []
    for face, embedding in current_faces:
        aluno_identificado = identify_face_in_group(reference_embeddings, embedding)
        if aluno_identificado:
            group.append(aluno_identificado)
    if len(group) > 1:
        logging.info(f"Grupo detectado: {group}")

# Função para identificar rostos em grupos
def identify_face_in_group(reference_embeddings, embedding, threshold=THRESHOLD):
    """Identifica um rosto dentro de um grupo."""
    best_similarity = 0
    best_aluno = None
    for aluno, ref_embeddings in reference_embeddings.items():
        for ref_embedding in ref_embeddings:
            similarity = cos_sim(ref_embedding, embedding)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_aluno = aluno
    return best_aluno

# Função para analisar emoções com DeepFace
def analyze_emotion(frame):
    """Analisa as emoções no rosto detectado."""
    result = DeepFace.analyze(frame, actions=['emotion'])
    return result['dominant_emotion']

# Inicialização do sistema
def main():
    initialize_directories()
    face_app = initialize_face_model()
    yolo_model = initialize_yolo_model()

    # Carregar embeddings existentes
    reference_embeddings = load_reference_embeddings(ALUNOS_DIR, face_app)

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduzir resolução para acelerar o processamento
        frame = reduce_resolution(frame, scale_percent=75)

        # Normalizar iluminação da imagem
        frame = normalize_lighting(frame)

        # Processar a cada N frames
        frame_count += 1
        if frame_count % FRAME_INTERVAL == 0:
            # Identificar e verificar duplicatas
            identify_and_check_duplicates_parallel(frame, face_app, yolo_model, reference_embeddings)
            
            # Verificar emoções
            emotion = analyze_emotion(frame)
            logging.info(f"Emoção detectada: {emotion}")

            # Verificar acessórios ou máscara facial
            accessories = detect_accessories(yolo_model, frame)
            if detect_mask(yolo_model, frame):
                logging.info("Máscara detectada, ajustando reconhecimento para focar nos olhos.")

        cv2.imshow("Verificação Multimodal de Indivíduos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
