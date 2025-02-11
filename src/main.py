from .camera import initialize_camera
from .directories import initialize_directories
from .face_model import initialize_face_model
from .yolo_model import initialize_yolo_model
from .fetch_photos import schedule_fetch_photos
from .process_frame import process_frame
from .utils import load_reference_embeddings
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor


def main():
    initialize_directories()
    face_app = initialize_face_model()
    yolo_model = initialize_yolo_model()

    threading.Thread(target=schedule_fetch_photos, daemon=True).start()

    cap = initialize_camera()

    executor = ThreadPoolExecutor(max_workers=2)
    frame_count = 0
    reference_embeddings = load_reference_embeddings(face_app)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        future = executor.submit(process_frame, frame, face_app, reference_embeddings)
        processed_frame = future.result()

        cv2.imshow("Verificação Multimodal de Indivíduos", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
