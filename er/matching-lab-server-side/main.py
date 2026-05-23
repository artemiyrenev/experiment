from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
from elastic_client import new_async_es_client
from config import load_config
from elastic_index import create_index_if_not_exists
from goods_rp import GoodsRepository
from logger import setup_json_logger

goods_repo: GoodsRepository = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global goods_repo
    
    config = load_config("./config.yaml")
    es_client = new_async_es_client(config)
    await create_index_if_not_exists(config, es_client)
    repo_logger = setup_json_logger("repository", config.log_level)
    goods_repo = GoodsRepository(es=es_client, index_name=config.elastic.index_name, log=repo_logger)
    
    yield
    
    await es_client.close()

app = FastAPI(title="Matching Lab API", lifespan=lifespan)

class UploadRequest(BaseModel):
    items: List[Dict[str, Any]]

@app.post("/upload", status_code=202)
async def upload_data(request: UploadRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(goods_repo.process_and_upload_batch, request.items)
    return {"message": f"Батч из {len(request.items)} записей принят в фоновую обработку"}

class MatchRequest(BaseModel):
    account_id: str
    queries: List[str]
    top_k: int = 30
    batch_size: int = 50
    # Новые параметры для тюнинга гиперпараметров
    rrf_weight: float = 0.1
    digit_penalty: float = 0.2

@app.post("/match")
async def bulk_match(request: MatchRequest):
    results = await goods_repo.experiment_search(
        account_id=request.account_id,
        queries=request.queries,
        top_k=request.top_k,
        batch_size=request.batch_size,
        rrf_weight=request.rrf_weight,
        digit_penalty=request.digit_penalty
    )
    return {"data": results}