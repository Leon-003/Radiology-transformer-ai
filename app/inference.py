import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "models/radiology_transformer"
)

def predict(inputs):

    predictions = model.predict(inputs)

    return predictions