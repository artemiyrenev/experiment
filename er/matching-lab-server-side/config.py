import sys
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class ElasticConfig:
    url: str
    index_name: str
    login: Optional[str] = None
    password: Optional[str] = None


class Config:
    def __init__(self, cfg: dict):
        try:
            # --- APP ---
            app = cfg["app"]
            self.app_url = app["APP_URL"]
            self.log_level = app.get("LOG_LEVEL", "info")

            # --- S3 (Minio) ---
            # Оставляем как было, предполагая, что S3 нужен
            if "s3" in cfg:
                s3 = cfg["s3"]
                self.s3_url = s3["S3_URL"]
                self.minio_access_key = s3["MINIO_ACCESS_KEY"]
                self.minio_secret_key = s3["MINIO_SECRET_KEY"]
                self.minio_bucket = s3["MINIO_BUCKET"]

            # --- SECRETS ---
            if "secrets" in cfg:
                secrets = cfg["secrets"]
                self.jwt = secrets["JWT"]
                self.val = secrets["VAL"]

            # --- ELASTIC ---
            elastic = cfg["elastic"]
            
            # Получаем значения. Если в YAML стоит "", то python получит пустую строку.
            # Если ключа нет совсем, получим None.
            login_val = elastic.get("Login")
            pass_val = elastic.get("Password")

            # Превращаем пустые строки в None для чистоты (не обязательно, но полезно)
            if not login_val:
                login_val = None
            if not pass_val:
                pass_val = None

            self.elastic = ElasticConfig(
                url=elastic["Elastic_URL"],
                login=login_val,
                password=pass_val,
                # Если IndexName не указан, ставим наш новый по умолчанию
                index_name=elastic.get("IndexName", "goods_index_v2"), 
            )

        except KeyError as e:
            sys.exit(f"Config error: missing field {e}")


def load_config(path: str) -> Config:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        sys.exit(f"Failed to read config file: {e}")

    return Config(data)