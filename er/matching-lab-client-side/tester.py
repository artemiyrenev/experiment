import os
import json
import httpx
import asyncio
import time

# Настройки
SERVER_URL = "http://213.172.7.235:6778/match"
DATASET_DIR = "../dataset/supply"
LOG_FILE = "er_report.log"

async def process_directory(dir_path: str, client: httpx.AsyncClient):
    folder_name = os.path.basename(dir_path)
    parsed_path = os.path.join(dir_path, "parsed.json")
    matched_path = os.path.join(dir_path, "matched.json")

    if not os.path.exists(parsed_path) or not os.path.exists(matched_path):
        return None

    try:
        with open(parsed_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        with open(matched_path, 'r', encoding='utf-8') as f:
            matched_data = json.load(f)
    except json.JSONDecodeError:
        return None

    queries = [item.get("name", "") for item in parsed_data if "name" in item]
    if not queries:
        return None

    # Пул доступных правильных ID для этой накладной
    available_valid_ids = [item.get("id") for item in matched_data if item.get("id")]

    request_payload = {
            "account_id": "12345",
            "queries": queries,
            "top_k": 50,        
            "rerank_limit": 30,
            "batch_size": 50
        }

    try:
        response = await client.post(SERVER_URL, json=request_payload, timeout=120.0)
        response.raise_for_status()
        server_results = response.json().get("data", [])
    except Exception:
        server_results = []

    correct = 0
    errors = []

    # Возвращаем твой оригинальный ЖАДНЫЙ поиск
    for i, res in enumerate(server_results):
        predicted_id = res.get("matched_id")
        query_name = queries[i] if i < len(queries) else "UNKNOWN"

        if predicted_id and predicted_id in available_valid_ids:
            correct += 1
            # Удаляем найденный ID из пула, чтобы не сматчить его дважды
            available_valid_ids.remove(predicted_id)
        else:
            errors.append({
                "query": query_name,
                "found": predicted_id
            })

    result_data = {
        "dir": folder_name,
        "total": len(queries),
        "correct": correct,
        "errors": errors,
        "missed_ids": available_valid_ids # Те ID, которые мы ожидали, но сервер их не выдал
    }

    if errors or available_valid_ids:
        result_data["full_request"] = queries
        result_data["full_expected"] = matched_data
        result_data["full_response"] = server_results

    return result_data

async def main():
    abs_dataset_dir = os.path.abspath(DATASET_DIR)
    
    if not os.path.exists(abs_dataset_dir):
        print(f"❌ Папка {abs_dataset_dir} не найдена.")
        return

    folders = sorted([f for f in os.listdir(abs_dataset_dir) if os.path.isdir(os.path.join(abs_dataset_dir, f))])
    all_results = []

    print("⏳ Выполняется матчинг, подожди...")
    start_time = time.time()

    async with httpx.AsyncClient(proxy=None, timeout=120.0) as client:
        for folder_name in folders:
            dir_path = os.path.join(abs_dataset_dir, folder_name)
            res = await process_directory(dir_path, client)
            if res:
                all_results.append(res)

    if not all_results:
        print("⚠️ Нет данных для анализа.")
        return

    total_queries = sum(r["total"] for r in all_results)
    total_correct = sum(r["correct"] for r in all_results)
    accuracy = (total_correct / total_queries * 100) if total_queries > 0 else 0
    elapsed = time.time() - start_time

    # Запись в единый файл
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=== ИТОГОВЫЙ ОТЧЕТ ===\n")
        f.write(f"Обработано накладных: {len(all_results)} из {len(folders)}\n")
        f.write(f"Всего позиций: {total_queries}\n")
        f.write(f"Успешных матчей: {total_correct}\n")
        f.write(f"Точность (Accuracy): {accuracy:.2f}%\n")
        f.write(f"Время выполнения: {elapsed:.2f} сек\n")
        f.write("======================\n\n")

        for r in all_results:
            if r["errors"] or r["missed_ids"]:
                f.write(f"{'='*60}\n")
                f.write(f"📁 НАКЛАДНАЯ: {r['dir']} (Найдено верно: {r['correct']} из {r['total']})\n")
                f.write(f"{'='*60}\n")
                
                if r["errors"]:
                    f.write("❌ ПРОМАХИ СЕРВЕРА (Запрос -> Неверный ID):\n")
                    for err in r["errors"]:
                        f.write(f"  [ЗАПРОС] {err['query']}\n")
                        f.write(f"  [НАШЛИ]  {err['found']}\n")
                        f.write(f"  {'-'*40}\n")
                
                if r["missed_ids"]:
                    f.write("\n⚠️ ОЖИДАЛИ, НО НЕ НАШЛИ (Оставшиеся правильные ID в пуле):\n")
                    for missed in r["missed_ids"]:
                        f.write(f"  - {missed}\n")
                    f.write(f"  {'-'*40}\n")

                f.write("\n📦 ПОЛНЫЙ ЗАПРОС (queries):\n")
                json.dump(r.get("full_request", []), f, ensure_ascii=False, indent=2)
                f.write("\n\n📦 ПОЛНЫЙ ОЖИДАЕМЫЙ ОТВЕТ (matched.json):\n")
                json.dump(r.get("full_expected", []), f, ensure_ascii=False, indent=2)
                f.write("\n\n📦 ПОЛНЫЙ ОТВЕТ СЕРВЕРА (server_results):\n")
                json.dump(r.get("full_response", []), f, ensure_ascii=False, indent=2)
                f.write("\n\n\n")

    print(f"✅ Готово! Обработано накладных: {len(all_results)}.")
    print(f"📈 Точность: {accuracy:.2f}% ({total_correct}/{total_queries}).")
    print(f"📝 Подробный лог сохранен в файл: {LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())