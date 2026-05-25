from transformers import (
    AutoTokenizer,
    TFAutoModel
)

import tensorflow as tf

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

def build_model(num_classes):

    encoder = TFAutoModel.from_pretrained(
        MODEL_NAME
    )

    input_ids = tf.keras.layers.Input(
        shape=(256,),
        dtype=tf.int32,
        name="input_ids"
    )

    attention_mask = tf.keras.layers.Input(
        shape=(256,),
        dtype=tf.int32,
        name="attention_mask"
    )

    embeddings = encoder(
        input_ids,
        attention_mask=attention_mask
    )[1]

    x = tf.keras.layers.Dropout(0.3)(
        embeddings
    )

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='sigmoid'
    )(x)

    model = tf.keras.Model(
        inputs=[
            input_ids,
            attention_mask
        ],
        outputs=outputs
    )

    return model