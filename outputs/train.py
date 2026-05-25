from model import build_model
from preprocess import tokenize_texts

import tensorflow as tf

NUM_CLASSES = 14

model = build_model(NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=2e-5
    ),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

model.save(
    "models/radiology_transformer"
)