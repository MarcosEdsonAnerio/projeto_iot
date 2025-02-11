import os
import cv2
import numpy as np
import requests
import logging
from sklearn.metrics.pairwise import cosine_similarity
from .config import THRESHOLD, BASE_URL, AULA_ID


def process_frame(frame, face_app, reference_embeddings):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        live_embedding = face.embedding

        best_similarity = 0
        best_aluno = "Desconhecido"
        best_matricula = None

        if live_embedding is not None and len(live_embedding) > 0:
            for aluno, ref_embeddings in reference_embeddings.items():
                if ref_embeddings and all(
                    isinstance(e, np.ndarray) and len(e) > 0 for e in ref_embeddings
                ):
                    try:
                        ref_embeddings_array = np.vstack(ref_embeddings)
                        similarity = max(
                            cosine_similarity([live_embedding], ref_embeddings_array)[0]
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_aluno = aluno
                            best_matricula = aluno.split("_")[-1]
                    except ValueError as e:
                        logging.error(
                            f"Erro ao calcular similaridade para {aluno}: {e}"
                        )
                        continue

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

        if best_similarity >= THRESHOLD and best_matricula:
            url = f"{BASE_URL}/ensino/diario/id_diario/aulas/frequencias"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('JWT_TOKEN')}",
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                frequencias = response.json()
                matricula_presente = any(
                    freq["matricula"]["id"] == best_matricula
                    and freq["frequencia"] == "P"
                    for freq in frequencias
                )
                if matricula_presente:
                    logging.info(
                        f"Matrícula {best_matricula} já presente com frequência 'P'"
                    )
                    continue

            data = {
                "aulas": [
                    {
                        "aula": {"id": AULA_ID},
                        "matricula": {"id": best_matricula},
                        "frequencia": "P",
                    }
                ]
            }

            url = f"{BASE_URL}/ensino/diario/aulas/frequencia"
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                logging.info(f"Requisição enviada com sucesso para {best_aluno}")
            else:
                logging.error(
                    f"Erro ao enviar requisição para {best_aluno}: {response.status_code}"
                )

    return frame
