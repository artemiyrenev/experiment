import json
import requests
from tqdm import tqdm

DATA_FILE = "data.json"
API_URL = "http://213.172.7.235:6778/upload"
BATCH_SIZE = 500

def chunk_data(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def main():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
    
    chunks = list(chunk_data(all_items, BATCH_SIZE))
    for batch in tqdm(chunks, desc="Отправка на er-server"):
        response = requests.post(API_URL, json={"items": batch})
        if response.status_code != 202:
            print(f"Ошибка: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()