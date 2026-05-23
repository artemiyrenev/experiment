FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libmagic1 libmagic-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# КЭШИРОВАННЫЕ СЛОИ С ТЯЖЕЛЫМИ ЗАВИСИМОСТЯМИ
# ==========================================

# Слой 1: PyTorch (Абсолютный тяжеловес, ~2 ГБ)
RUN pip install --no-cache-dir torch==2.10.0

# Слой 2: ONNX Runtime для GPU (~300-400 МБ)
RUN pip install --no-cache-dir onnxruntime-gpu==1.23.2

# Слой 3: Модели, NLP и экосистема HuggingFace (~200-300 МБ)
RUN pip install --no-cache-dir transformers==5.3.0 sentence-transformers==5.3.0

# Слой 4: Научные и математические библиотеки (~250 МБ)
RUN pip install --no-cache-dir scipy==1.15.3 scikit-learn==1.7.2 numpy==2.2.6

# ==========================================
# СЛОЙ С ОСТАЛЬНЫМИ (ЛЕГКИМИ) ПАКЕТАМИ
# ==========================================

COPY requirements.txt .
# pip увидит, что тяжелые библиотеки уже установлены, 
# скажет "Requirement already satisfied" и быстро доустановит остальную мелочь (FastAPI, uvicorn и т.д.)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код в самом конце, так как он меняется чаще всего
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
