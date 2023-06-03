import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from xgboost import XGBClassifier


def predict(image):
    # __________________CREATE AND LOAD MODELS_______________
    model = XGBClassifier()
    model.load_model("model_xgb.h5")
    model2 = keras.models.load_model("model2.h5")
    new_model = keras.models.Model(
        model2.input, model2.get_layer("global_average_pooling2d").output
    )

    # _________________CONVERT TO DATAFRAME AND MAKE PREDICTIONS________________
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img_array = cv2.resize(decoded, (224, 224))
    data = pd.DataFrame({"image": [img_array], "label": [0]})
    df = pd.DataFrame()
    df = pd.concat([df, data])
    test_x = np.array(df.image.to_list())
    X_test_features = new_model.predict(test_x)
    y_pred = model.predict_proba(X_test_features)
    labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]
    result = {
        labels[0]: round(float(y_pred[0][0]) * 100, 2),
        labels[1]: round(float(y_pred[0][1]) * 100, 2),
        labels[2]: round(float(y_pred[0][2]) * 100, 2),
        labels[3]: round(float(y_pred[0][3]) * 100, 2),
    }
    return result
