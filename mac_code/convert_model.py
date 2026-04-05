import tensorflow as tf
import emlearn

# 1. Load the Keras model we just trained
print("Loading prs_model.h5...")
model = tf.keras.models.load_model('prs_model.h5')

# 2. Convert the model to C/MicroPython compatible format using emlearn
print("Converting model to ESP32 format...")
cmodel = emlearn.keras.convert(model)

# 3. Save it as a .tmdl file
output_filename = 'prs_cnn.tmdl'
cmodel.save(file=output_filename)

print(f"\n✅ Success! Model converted and saved as {output_filename}")
print("You are ready to upload this to the ESP32!")