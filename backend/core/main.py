from core.predict import predict
from fastapi import FastAPI, File, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pympler import asizeof

app = FastAPI()

origins = [
    "http://react",
    "http://react:3000" "http://react:81",
    "http://react:80",
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
    if not file:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "No file selected."},
        )

    if file[0:3] != b"\xff\xd8\xff" and file[0:3] != b"\x89\x50\x4e":
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "message": "It seems like file you've sent is not an image. Only JPEG/PNG images are accepted."
            },
        )

    if asizeof.asizeof(file) / (1024 * 1024) >= 64:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "message": "Your file is too big. Ensure the file size is less than 64M"
            },
        )
    try:
        predictions = predict(file)
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"result": predictions}
        )
    except ValueError as err:
        print(err)  # LOGGING
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Something went wrong. "},
        )
