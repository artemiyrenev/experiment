import asyncio
import httpx
import os
import json
from itertools import product

# Убедись, что путь к датасету верный
SERVER_URL = "http://213.172.7.235:6778/match"
DATASET_DIR = "../dataset/supply"

# Сетка гиперпараметров для подбора
grid = {
    "rrf_weight": [0.05, 0.2, 0.3, 0.4],
    "digit_penalty": [0.01, 0.1, 0.2, 0.4, 0.5]
}

async def run_iteration(client, params, folders):
    total_queries, total_correct = 0, 0
    
    for folder_name in folders:
        dir_path = os.path.join(DATASET_DIR, folder_name)
        parsed_path = os.path.join(dir_path, "parsed.json")
        matched_path = os.path.join(dir_path, "matched.json")
        
        if not (os.path.exists(parsed_path) and os.path.exists(matched_path)):
            continue
            
        with open(parsed_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        with open(matched_path, 'r', encoding='utf-8') as f:
            matched_data = json.load(f)

        queries = [item.get("name", "") for item in parsed_data if "name" in item]
        if not queries:
            continue

        available_valid_ids = [item.get("id") for item in matched_data if item.get("id")]
        
        request_payload = {
            "account_id": "12345",
            "queries": queries,
            "top_k": 30,
            "rrf_weight": params["rrf_weight"],
            "digit_penalty": params["digit_penalty"]
        }

        try:
            resp = await client.post(SERVER_URL, json=request_payload, timeout=120.0)
            server_results = resp.json().get("data", [])
            
            # Жадная логика проверки совпадений
            for res in server_results:
                predicted_id = res.get("matched_id")
                if predicted_id and predicted_id in available_valid_ids:
                    total_correct += 1
                    available_valid_ids.remove(predicted_id)
                    
            total_queries += len(queries)
        except Exception as e:
            continue

    accuracy = (total_correct / total_queries * 100) if total_queries > 0 else 0
    return accuracy

async def main():
    abs_dataset_dir = os.path.abspath(DATASET_DIR)
    if not os.path.exists(abs_dataset_dir):
        print(f"❌ Папка {abs_dataset_dir} не найдена.")
        return

    folders = sorted([f for f in os.listdir(abs_dataset_dir) if os.path.isdir(os.path.join(abs_dataset_dir, f))])
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"🚀 Запуск Grid Search. Всего комбинаций: {len(combinations)}")
    print("-" * 50)
    
    results = []

    async with httpx.AsyncClient(proxy=None) as client:
        for params in combinations:
            acc = await run_iteration(client, params, folders)
            results.append((acc, params))
            print(f"📊 rrf_weight: {params['rrf_weight']}, digit_penalty: {params['digit_penalty']} -> Accuracy: {acc:.2f}%")

    results.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_params = results[0]

    print("\n" + "="*50)
    print(f"🏆 ЛУЧШАЯ ТОЧНОСТЬ: {best_acc:.2f}%")
    print(f"🛠 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ: {best_params}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())