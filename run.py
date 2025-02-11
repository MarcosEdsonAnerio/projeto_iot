from src.main import main
from src.auth import set_jwt_token

if __name__ == "__main__":
    set_jwt_token()
    main()
