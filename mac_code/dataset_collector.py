# ---------------------------------------------------------
# ESP32 Project: AI Rock-Paper-Scissors
# Author: Dhruv Rawat
# 
# Citations/Collaborations:
# - Base camera/networking logic provided by ENMGT 5400 course materials.
# - AI debugging, OpenCV threading logic, and model architecture 
#   optimizations developed with the assistance of Google Gemini.
# ---------------------------------------------------------
import cv2
import os

# --- Configuration ---
# REPLACE THE IP WITH YOUR ESP32'S IP ADDRESS
ESP32_URL = 'http://172.20.10.2:8080/xiao/Hi-Xiao-Ling'

# Create folders for your dataset
folders = ['dataset/rock', 'dataset/paper', 'dataset/scissors']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

counters = {'r': 0, 'p': 0, 's': 0}

print("Connecting to ESP32 Stream...")
vid = cv2.VideoCapture(ESP32_URL)

if not vid.isOpened():
    print("Failed to connect. Check the IP address and ensure the ESP32 server is running.")
    exit()

print("Connected!")
print("CONTROLS:")
print("Press 'r' to save a ROCK image")
print("Press 'p' to save a PAPER image")
print("Press 's' to save a SCISSORS image")
print("Press 'q' to QUIT")

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        continue

    # Show the live feed from the ESP32
    cv2.imshow('Live Stream', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # Process and save the image based on key press
    gesture = None
    if key == ord('r'):
        gesture = 'rock'
        key_char = 'r'
    elif key == ord('p'):
        gesture = 'paper'
        key_char = 'p'
    elif key == ord('s'):
        gesture = 'scissors'
        key_char = 's'

    if gesture:
        # 1. Convert to Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to 32x32 (Professor's required training size)
        # We use INTER_AREA because we are shrinking the image
        small_frame = cv2.resize(gray_frame, (32, 32), interpolation=cv2.INTER_AREA)
        
        # 3. Save as BMP
        filename = f"dataset/{gesture}/{gesture}_{counters[key_char]:04d}.bmp"
        cv2.imwrite(filename, small_frame)
        print(f"Saved: {filename}")
        
        counters[key_char] += 1

vid.release()
cv2.destroyAllWindows()
