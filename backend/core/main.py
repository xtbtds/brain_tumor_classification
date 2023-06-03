from core.predict import predict
from fastapi import FastAPI, File, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def receiveFile(file: bytes = File(...)):
    try:
        predictions = predict(file)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"result": predictions})
    except ValueError as err:
        print(err)  # LOGGING
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "error"})
