import os
import math
import psutil
import re
import asyncio
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from elasticsearch import AsyncElasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

class HardwareOptimizer:
    @staticmethod
    def configure():
        try:
            if os.path.exists("/sys/fs/cgroup/cpu.max"):
                with open("/sys/fs/cgroup/cpu.max") as f:
                    quota, period = f.read().split()
                    if quota != "max":
                        real_cores = int(math.ceil(int(quota) / int(period)))
                    else:
                        real_cores = psutil.cpu_count(logical=False) or 2
            elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
                with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f_q, \
                     open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f_p:
                    quota, period = int(f_q.read()), int(f_p.read())
                    if quota > 0:
                        real_cores = int(math.ceil(quota / period))
                    else:
                        real_cores = psutil.cpu_count(logical=False) or 2
            else:
                real_cores = psutil.cpu_count(logical=False) or 2
        except Exception:
            real_cores = psutil.cpu_count(logical=False) or 2

        optimal_threads = min(real_cores, 8)
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        return optimal_threads

OPTIMAL_THREADS = HardwareOptimizer.configure()
torch.set_num_threads(OPTIMAL_THREADS)

ml_executor = ThreadPoolExecutor(max_workers=1)

class GoodsRepository:
    def __init__(self, es: AsyncElasticsearch, index_name: str, log: logging.Logger):
        self.es = es
        self.index_name = index_name
        self.logger = log
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"🚀 Hardware Config: {OPTIMAL_THREADS} CPU threads. ML Device: {self.device.upper()}")
        
        self.logger.info("Загрузка BAAI/bge-m3 (Bi-Encoder)...")
        self.bi_encoder = SentenceTransformer('BAAI/bge-m3', device=self.device)
        
        self.model_repo = "BAAI/bge-reranker-v2-m3"
        base_cache_dir = os.getenv("MODEL_DIR", "/app/cache")
        self.cache_dir = os.path.join(base_cache_dir, "bge_reranker")
        
        model_exists = os.path.exists(self.cache_dir) and any(
            f.endswith('.safetensors') or f.endswith('.bin') for f in os.listdir(self.cache_dir)
        )
        
        if not model_exists:
            self.logger.info(f"⬇️ Модель реранкера не найдена локально. Скачиваем {self.model_repo} в {self.cache_dir}...")
            os.makedirs(self.cache_dir, exist_ok=True)
            snapshot_download(
                repo_id=self.model_repo,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.onnx", "*.h5", "*.msgpack", "*.ot"]
            )
            self.logger.info("✅ Скачивание реранкера завершено!")
        else:
            self.logger.info(f"✅ Реранкер найден в кэше: {self.cache_dir}. Пропускаем скачивание.")

        self.logger.info("Загрузка реранкера в память...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.cache_dir)
        
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.cache_dir,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.rerank_model.eval()

    async def _encode_async(self, texts: List[str]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        clean_texts = [str(t) if t else "" for t in texts]
        return await loop.run_in_executor(
            ml_executor, 
            lambda: self.bi_encoder.encode(clean_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
        )

    def _torch_predict(self, pairs: List[List[str]]) -> Tuple[np.ndarray, float]:
        if not pairs: return np.array([]), 0.0
        clean_pairs = [[str(p[0] or ""), str(p[1] or "")] for p in pairs]
        
        try:
            features = self.rerank_tokenizer(
                clean_pairs, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.rerank_model(**features)
                logits = outputs.logits.view(-1).float().cpu().numpy()
            
            scores = 1 / (1 + np.exp(-np.clip(logits, -15, 15)))
            return scores, 0.0
            
        except Exception:
            self.logger.exception("💥 PyTorch Inference Failed") 
            return np.zeros(len(pairs)), 0.0

    async def _rerank_smart_async(self, pairs: List[List[str]]) -> Tuple[np.ndarray, float]:
        if not pairs: return np.array([]), 0.0
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(ml_executor, lambda: self._torch_predict(pairs))

    def _extract_digit_penalty(self, query: str, name: str, penalty_value: float) -> float:
        pattern = r'(?<!\d)\d+(?:[\.,]\d+)?(?!\d)'
        q_nums = set(n.replace(',', '.') for n in re.findall(pattern, query))
        if not q_nums: return 1.0
        n_nums = set(n.replace(',', '.') for n in re.findall(pattern, name))
        if not q_nums.issubset(n_nums): return penalty_value
        return 1.0

    async def experiment_search(
        self, account_id: str, queries: List[str], top_k: int = 30, batch_size: int = 50, 
        rerank_limit: int = 15, rrf_weight: float = 0.1, digit_penalty: float = 0.2, **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        if not queries: return results

        with tqdm(total=len(queries), desc="🚀 Search Pipeline", unit="query") as pbar:
            for start in range(0, len(queries), batch_size):
                chunk_queries = queries[start : start + batch_size]
                batch_results = [{"query": q, "matched_id": None, "top_score": 0.0, "candidates": []} for q in chunk_queries]

                try:
                    query_vectors = await self._encode_async(chunk_queries)
                    qv_list = query_vectors.tolist()

                    body = []
                    for i, q_text in enumerate(chunk_queries):
                        filters = [{"term": {"account_id": account_id}}, {"term": {"product.archived": False}}]
                        body.extend([
                            {"index": self.index_name},
                            {"size": top_k, "query": {"bool": {"should": [{"match": {"product.name": {"query": q_text, "boost": 3}}}, {"match": {"product.name.partial": {"query": q_text}}}], "filter": filters}}},
                            {"index": self.index_name},
                            {"size": top_k, "knn": {"field": "product_vector", "query_vector": qv_list[i], "k": top_k, "num_candidates": 100, "filter": filters}}
                        ])
                    
                    resp = await asyncio.wait_for(self.es.msearch(body=body), timeout=15.0)
                    responses = resp.get("responses", [])
                    flat_rerank_queue = [] 

                    for i in range(len(chunk_queries)):
                        b_hits = responses[i*2].get("hits", {}).get("hits", []) if "error" not in responses[i*2] else []
                        k_hits = responses[i*2+1].get("hits", {}).get("hits", []) if "error" not in responses[i*2+1] else []
                        
                        b_ranks = {h["_id"]: r for r, h in enumerate(b_hits, 1)}
                        k_ranks = {h["_id"]: r for r, h in enumerate(k_hits, 1)}
                        all_ids = set(b_ranks.keys()) | set(k_ranks.keys())
                        
                        max_b = b_hits[0]["_score"] if b_hits else 0.001
                        max_k = k_hits[0]["_score"] if k_hits else 0.001

                        bm25_threshold = max_b * 0.45 
                        knn_threshold = max_k - 0.12  

                        b_dict = {h["_id"]: h for h in b_hits}
                        k_dict = {h["_id"]: h for h in k_hits}

                        candidates = []
                        for d_id in all_ids:
                            h_l = b_dict.get(d_id)
                            h_s = k_dict.get(d_id)
                            score_b = h_l["_score"] if h_l else 0.0
                            score_k = h_s["_score"] if h_s else 0.0
                            raw_source = (h_l or h_s).get("_source", {})
                            prod_name = raw_source.get("product", {}).get("name", "")
                            
                            if not prod_name: continue
                            
                            if score_b >= bm25_threshold or score_k >= knn_threshold:
                                rrf = (1.0 / (60 + b_ranks.get(d_id, 1000))) + (1.0 / (60 + k_ranks.get(d_id, 1000)))
                                candidates.append({
                                    "id": d_id, "name": prod_name, "rrf_score": rrf, "raw_ce_score": 0.0, "score": 0.0
                                })
                        
                        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
                        top_candidates = candidates[:rerank_limit]
                        
                        batch_results[i]["candidates"] = top_candidates
                        for cand in top_candidates:
                            flat_rerank_queue.append((i, cand))

                    if flat_rerank_queue:
                        pairs = [[chunk_queries[idx], cand["name"]] for idx, cand in flat_rerank_queue]
                        flat_scores, _ = await self._rerank_smart_async(pairs)
                        for k, score in enumerate(flat_scores):
                            _, cand = flat_rerank_queue[k]
                            cand["raw_ce_score"] = float(score)

                    for i, q_text in enumerate(chunk_queries):
                        active_cands = batch_results[i]["candidates"]
                        if not active_cands: continue

                        max_r = max(c["rrf_score"] for c in active_cands)
                        min_r = min(c["rrf_score"] for c in active_cands)

                        for c in active_cands:
                            norm_rrf = (c["rrf_score"] - min_r) / (max_r - min_r + 1e-9) if max_r > min_r else 1.0
                            
                            # Применяем переданные параметры штрафа
                            penalty = self._extract_digit_penalty(q_text, c["name"], penalty_value=digit_penalty)
                            
                            # Применяем баланс rrf_weight
                            c["score"] = c["raw_ce_score"] * ((1.0 - rrf_weight) + rrf_weight * norm_rrf) * penalty

                        active_cands.sort(key=lambda x: x["score"], reverse=True)
                        best = active_cands[0]
                        batch_results[i]["matched_id"] = best["id"]
                        batch_results[i]["top_score"] = best["score"]
                    
                    results.extend(batch_results)
                    pbar.update(len(chunk_queries))

                except Exception:
                    self.logger.exception(f"💥 Batch Error")
                    for res in batch_results: res["error"] = "Pipeline failure"
                    results.extend(batch_results)
                    pbar.update(len(chunk_queries))

        return results

    async def process_and_upload_batch(self, items: List[Dict[str, Any]]):
        if not items:
            return

        self.logger.info(f"Начало обработки батча из {len(items)} записей...")
        names = [item.get('name', '') for item in items]
        
        vectors = await self._encode_async(names)
        vectors_f32 = vectors.astype(np.float32)

        async def document_generator():
            for i, item in enumerate(items):
                doc = {
                    "_index": self.index_name,
                    "_id": item.get('uuid'),
                    "_source": {
                        "account_id": str(item.get('account_id', '12345')),
                        "product_vector": vectors_f32[i].tolist(),
                        "product": {
                            "id": item.get('uuid'),
                            "name": item.get('name', ''),
                            "archived": bool(item.get('archived', False))
                        }
                    }
                }
                if 'updated' in item:
                    doc["_source"]["product"]["updated"] = item['updated']
                yield doc

        try:
            await helpers.async_bulk(
                self.es, 
                document_generator(),
                chunk_size=500,
                raise_on_error=True
            )
            self.logger.info("Батч успешно сохранен в БД.")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке батча: {e}")
            raise e