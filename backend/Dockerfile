FROM python:3.10.9-slim-bullseye
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install python-multipart
EXPOSE 8080
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]