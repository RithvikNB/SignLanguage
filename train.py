import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, verbose=0
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, verbose=0
)

class_names = train_ds.class_names
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

autotune = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=autotune)
val_ds = val_ds.prefetch(buffer_size=autotune)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
])

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=4, verbose=0)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy")
model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[early_stop], verbose=0)

model.save("asl_mobilenetv2.keras")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"Final Validation Accuracy: {accuracy:.4f}")
