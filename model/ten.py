import tensorflow as tf

model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
lite_model = converter.convert()

with open("efficientnet_lite.tflite", "wb") as f:
    f.write(lite_model)
