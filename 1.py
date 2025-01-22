import tensorflow as tf

# Define input shape
model = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
lite_model = converter.convert()

# Save the converted model
with open("models/efficientnet_lite.tflite", "wb") as f:
    f.write(lite_model)
