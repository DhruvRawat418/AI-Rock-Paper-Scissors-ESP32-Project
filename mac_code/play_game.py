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
import numpy as np
import tensorflow as tf
import threading

# --- 1. THE LAG FIX: Background Video Reader ---
class LiveStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Try to force the buffer size to 1
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        # Start a background process that constantly eats frames
        # so the main AI loop always gets the absolute newest image.
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.running = False
        self.cap.release()

# --- 2. Load Model ---
print("Loading Keras Model...")
model = tf.keras.models.load_model('prs_model.h5')
CLASSES = ['paper', 'rock', 'scissors']

ESP32_URL = 'http://192.168.4.1:8080/xiao/Hi-Xiao-Ling'
print(f"Connecting to live video stream at {ESP32_URL}...")

# Use our new lag-proof stream!
stream = LiveStream(ESP32_URL)

if not stream.ret:
    print("Error: Could not connect to the stream.")
    exit()

print("\n✅ Lag-Proof AI Inference Started!")

# --- 3. Main Game Loop ---
while True:
    # This now instantly grabs the freshest frame, bypassing the waiting line
    ret, img = stream.read()
    
    if not ret or img is None:
        continue
        
    # --- AI PREPROCESSING ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_32 = cv2.resize(gray, (32, 32))
    
    input_data = resized_32 / 255.0
    input_data = input_data.reshape(1, 32, 32, 1)
    
    # --- AI INFERENCE ---
    predictions = model.predict(input_data, verbose=0)[0]
    max_idx = np.argmax(predictions)
    confidence = predictions[max_idx]
    
    # --- DISPLAY DASHBOARDS ---
    if confidence > 0.70:
        label = f"{CLASSES[max_idx].upper()} ({confidence*100:.0f}%)"
        color = (0, 255, 0)
    else:
        label = f"Guessing: {CLASSES[max_idx].upper()}..."
        color = (0, 165, 255)
        
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Main Window
    display_img = cv2.resize(img, (400, 400))
    cv2.imshow('Live Rock Paper Scissors AI', display_img)
    
    # Debug Window (What the AI Sees)
    ai_vision = cv2.resize(resized_32, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('What the AI Sees (32x32)', ai_vision)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
