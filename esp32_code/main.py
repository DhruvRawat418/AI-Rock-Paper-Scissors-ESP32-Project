"""
main.py - Rock Paper Scissors CNN Inference on XIAO ESP32-S3 Sense
Runs on the ESP32. Captures camera images, preprocesses them,
and classifies them as rock, paper, or scissors using emlearn CNN.

Author: [Your Name]
Based on professor's test_tmdl_from_camera.py template and
image_preprocessing.py provided by course staff.
"""

import array
import gc
from camera import Camera, PixelFormat, FrameSize
import emlearn_cnn_fp32 as emlearn_cnn
from image_preprocessing import resize_96x96_to_32x32_and_threshold, strip_bmp_header

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the trained model file (must be uploaded to ESP)
MODEL = 'prs_cnn.tmdl'

# Confidence threshold: predictions below this are ignored as uncertain
RECOGNITION_THRESHOLD = 0.74

# Threshold for converting grayscale pixels to black/white before inference.
# Pixels >= this value become white (255), below become black (0).
# Tune this if classification accuracy is poor (try 100-180).
IMAGE_THRESHOLD = 128

# Camera pin configuration for XIAO ESP32-S3 Sense (do not change)
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
    "frame_size": FrameSize.R96X96,       # Capture at 96x96 (smallest native size)
    "pixel_format": PixelFormat.GRAYSCALE  # Grayscale saves memory vs colour
}

# Class labels - must match the order your model was trained with
CLASSES = ['paper', 'rock', 'scissors']

# ── Helper Functions ───────────────────────────────────────────────────────────

def argmax(values):
    """Returns the index of the highest value in a list.
    Used to pick the most confident class prediction."""
    max_val = values[0]
    max_idx = 0
    for i in range(1, len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
    return max_idx

def print_probabilities(classes, probs):
    """Prints each class with its confidence percentage. Useful for debugging."""
    print("Probabilities:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {probs[i]*100:.1f}%")

# ── Setup ──────────────────────────────────────────────────────────────────────

print("Initializing camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)  # Output as BMP so image_preprocessing functions work
print("Camera ready.")

# Load the model from file into memory
print(f"Loading model from {MODEL}...")
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())

gc.collect()  # Free up memory after loading

model = emlearn_cnn.new(model_data)
print("Model loaded and ready.")

# ── Output buffer for CNN predictions ─────────────────────────────────────────
# One probability output per class
n_classes = len(CLASSES)
probabilities = array.array('f', [0.0] * n_classes)

# Track the last prediction so we only print when it changes
current_prediction = 'none'
image_count = 0

# ── Main Inference Loop ────────────────────────────────────────────────────────
print("\nStarting inference loop. Point camera at your hand gesture.")
print("Press Ctrl+C to stop.\n")

while True:
    try:
        # Step 1: Capture a raw 96x96 grayscale BMP from the camera
        raw_image = cam.capture()
        image_count += 1

        # Step 2: Resize from 96x96 → 32x32 and apply threshold
        # This converts the image to a simple black/white 32x32 bitmap
        resized = resize_96x96_to_32x32_and_threshold(raw_image, IMAGE_THRESHOLD)

        # Step 3: Strip the BMP file header to get raw pixel bytes only
        # The CNN model expects only pixel data, not BMP metadata
        pixel_data = strip_bmp_header(resized)

        # Step 4: Run the CNN model on the pixel data
        model.run(pixel_data, probabilities)

        # Step 5: Find which class has the highest probability
        best_idx = argmax(probabilities)
        best_confidence = probabilities[best_idx]
        best_class = CLASSES[best_idx]

        # Step 6: Only print a result if confidence is above the threshold
        if best_confidence >= RECOGNITION_THRESHOLD:
            new_prediction = best_class
        else:
            new_prediction = 'uncertain'

        # Print only when the prediction changes (reduces output noise)
        if new_prediction != current_prediction:
            current_prediction = new_prediction
            if new_prediction == 'uncertain':
                print(f"[{image_count}] Uncertain (best: {best_class} {best_confidence*100:.0f}%)")
            else:
                print(f"[{image_count}] >>> {new_prediction.upper()} ({best_confidence*100:.0f}%)")
                print_probabilities(CLASSES, probabilities)

        # Free memory after each frame to prevent crashes
        del raw_image, resized, pixel_data
        gc.collect()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
    except Exception as e:
        print(f"Error on frame {image_count}: {e}")
        gc.collect()
        continue
