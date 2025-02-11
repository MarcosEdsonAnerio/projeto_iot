from insightface.app import FaceAnalysis
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if DEVICE.type == "cuda" else -1, det_size=(640, 640))
    return app
