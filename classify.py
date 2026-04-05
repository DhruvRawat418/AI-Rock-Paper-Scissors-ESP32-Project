import gc
import array
import time
import uctypes
from camera import Camera, PixelFormat, FrameSize
from image_preprocessing import resize_96x96_to_32x32_and_threshold, strip_bmp_header

import emlearn_cnn_fp32 as emlearn_cnn

MODEL = 'prs_cnn.tmdl'
CONFIDENCE_THRESHOLD = 0.70

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

def argmax(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx

# --- Initialization ---
print("Initializing Camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)

print("Loading Model into Memory...")
with open(MODEL, 'rb') as f:
    raw_data = f.read()

# Pad the file size to be a multiple of 16
pad_len = (16 - (len(raw_data) % 16)) % 16
if pad_len != 0:
    raw_data += b'\x00' * pad_len

print("Playing RAM Roulette for 16-byte alignment...")
aligned_model_data = None
garbage_collection_blocker = [] 

for i in range(100):
    candidate = array.array('B', raw_data)
    addr = uctypes.addressof(candidate)
    
    if addr % 16 == 0:
        print(f"✅ Won RAM Roulette! Address perfectly aligned: {hex(addr)}")
        aligned_model_data = candidate
        break
    else:
        garbage_collection_blocker.append(candidate)

if aligned_model_data is None:
    print("❌ Critical Error: Could not find an aligned memory block. Press RESET.")

del garbage_collection_blocker
del raw_data
gc.collect()

model = emlearn_cnn.new(aligned_model_data)
print("Model Initialized!")

# Alphabetical order for the 3 classes
classes = ['paper', 'rock', 'scissors']
probabilities = array.array('f', [0.0] * 3)

locked_until = 0
hand_was_away = True
recent = []

print("\n✅ Ready! Show me a gesture...")

# --- Main Inference Loop ---
while True:
    try:
        img = cam.capture()
        if not img:
            continue
            
        small = resize_96x96_to_32x32_and_threshold(img, 128)
        raw = strip_bmp_header(small)
        
        # --- THE MISSING LINK ---
        # Your friend's test script proved it MUST be an array.array('B')
        input_data = array.array('B', raw)
        
        # Pass both strictly-typed arrays to the AI
        model.run(input_data, probabilities)
        
        now = time.time()
        confidence = max(probabilities)
        prediction = classes[argmax(probabilities)]
        
        # Smoothing logic
        sorted_probs = sorted(list(probabilities))
        second_best = sorted_probs[-2]
        margin = confidence - second_best
        
        if margin < 0.10:
            hand_was_away = True
            recent = []
            continue
            
        if now < locked_until:
            continue
            
        if not hand_was_away:
            continue
            
        if confidence >= CONFIDENCE_THRESHOLD:
            recent.append(prediction)
        else:
            recent = []
            
        if len(recent) >= 3 and len(set(recent)) == 1:
            print(f">>> {prediction.upper()} (Confidence: {confidence*100:.0f}%)")
            locked_until = now + 3
            hand_was_away = False
            recent = []
            
    except KeyboardInterrupt:
        print("Stopped.")
        break
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")
        time.sleep(1)