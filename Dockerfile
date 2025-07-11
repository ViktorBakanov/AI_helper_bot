FROM python:3.10-slim

# 🧰 Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    python3-dev \
    libc6-dev \
    g++ \
    git \
    curl \
    wget \
    libstdc++6 \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# 🐍 Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .

# Обновляем pip, setuptools и wheel
RUN pip install --upgrade pip setuptools wheel

# Устанавливаем llama-cpp-python (соберётся из исходников)
RUN pip install llama-cpp-python

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 📦 Копируем всё остальное
COPY . .

# ⚡ Указываем команду по умолчанию
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]