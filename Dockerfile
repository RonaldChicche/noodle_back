FROM python:3.9.20-slim-bullseye

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -q git+https://github.com/THU-MIG/yolov10.git

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . . 

EXPOSE 5000

CMD ["python", "server.py"]
