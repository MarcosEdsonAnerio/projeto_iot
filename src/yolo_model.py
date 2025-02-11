from ultralytics import YOLO
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_yolo_model():
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
    return model
