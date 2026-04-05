import tensorflow as tf

print("Loading prs_model.h5...")
model = tf.keras.models.load_model('prs_model.h5')

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('prs_model.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("✅ Saved prs_model.tflite successfully!")