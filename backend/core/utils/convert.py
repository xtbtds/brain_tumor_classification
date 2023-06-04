import cv2
import numpy as np
import pandas as pd


def convert_image_to_dataframe(image: bytes) -> pd.DataFrame:
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img_array = cv2.resize(decoded, (224, 224))
    data = pd.DataFrame({"image": [img_array], "label": [0]})
    df = pd.DataFrame()
    df = pd.concat([df, data])
    return df
