# ---------------------------------------------------------
# ESP32 Project: Rock-Paper-Scissors
# Author: Dhruv Rawat
# 
# Citations/Collaborations:
# - Base camera/networking logic provided by ENMGT 5400 course materials.
# - AI debugging, OpenCV threading logic, and model architecture 
#   optimizations developed with the assistance of Google Gemini.
# ---------------------------------------------------------
import gc
import array
import time
from camera import Camera, PixelFormat, FrameSize
from image_preprocessing import resize_96x96_to_32x32_and_threshold, strip_bmp_header

# NOTE: You MUST have the emlearn_cnn_fp32 module on your ESP32 for this to work!
import emlearn_cnn_fp32 as emlearn_cnn

MODEL = 'prs_cnn.tmdl'
RECOGNITION_THRESHOLD = 0.74

CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# --- Initialization ---
print("Initializing Camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)

print("Loading Model...")
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())
    print("Model Data Loaded..")
    gc.collect()
    model = emlearn_cnn.new(model_data)
    print("Model Loaded..")

# Set up classes and tracking variables
classes = ['none', 'paper', 'rock', 'scissors']
current_prediction = classes[0]
cnt = 0

print("\n✅ Ready! Show me a gesture...")

# --- Main Loop ---
while True:
    try:
        # 1. Capture the image
        buf = cam.capture()
        if not buf:
            continue
            
        # 2. Resize and threshold
        processed_img = resize_96x96_to_32x32_and_threshold(buf)
        
        # 3. Strip headers to get binary data
        binary_data = strip_bmp_header(processed_img)
        
        # 4. Run model on binary data
        probabilities = model.run(binary_data)
        out = argmax(probabilities)
        confidence = probabilities[out]
        
        # 5. Check if prediction changed and print
        if confidence >= RECOGNITION_THRESHOLD:
            new_prediction = classes[out]
            if new_prediction != current_prediction:
                print(f">>> {new_prediction.upper()} (Confidence: {confidence*100:.1f}%)")
                current_prediction = new_prediction
                
        cnt += 1
        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(1)
