from typing import Dict

import cv2
import numpy as np
import pandas as pd
from core.utils.convert import convert_image_to_dataframe
from tensorflow import keras
from xgboost import XGBClassifier


def predict(image: bytes) -> Dict:
    # load XGBoost model
    model_xgb = XGBClassifier()
    model_xgb.load_model("model_xgb.h5")

    # load CNN model
    model_cnn = keras.models.load_model("model2.h5")

    # Get the output of GlobalAveragePooling layer
    new_model = keras.models.Model(
        model_cnn.input, model_cnn.get_layer("global_average_pooling2d").output
    )

    # convert image to dataframe
    df = convert_image_to_dataframe(image)

    # make predictions
    test_x = np.array(df.image.to_list())
    X_test_features = new_model.predict(test_x)
    y_pred = model_xgb.predict_proba(X_test_features)
    labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]
    result = {
        labels[i]: round(float(y_pred[0][i]) * 100, 2) for i in range(len(labels))
    }
    return result
