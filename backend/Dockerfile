FROM python:3.10.9-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

# install the cv2 dependencies into docker container
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install python-multipart
RUN pip3 install --no-cache-dir tensorflow

EXPOSE 8080

CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]