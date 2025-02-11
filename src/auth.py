import subprocess
import requests
import os
import json
from .config import BASE_URL, USER, PASSWORD


def get_jwt_token():
    url = f"{BASE_URL}/auth/signin"

    payload = json.dumps({"username": "administrador", "password": "proinfe@123"})
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 201:
        return response.json()["accessToken"]
    else:
        raise Exception("Erro ao buscar token JWT")


def set_jwt_token():
    token = get_jwt_token()
    os.environ["JWT_TOKEN"] = token
    print("JWT token set successfully")
