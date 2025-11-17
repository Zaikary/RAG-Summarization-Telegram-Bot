FROM python:3.11-slim

WORKDIR /app

# Устанавливаем только необходимые системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости сначала для лучшего кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY bot_main.py handlers.py indexer.py llm_handler.py ./
COPY db.txt ./

# Создаём директорию для индекса
RUN mkdir -p faiss_index

# Очищаем кэш pip
RUN pip cache purge

# Запускаем
CMD ["python", "bot_main.py"]