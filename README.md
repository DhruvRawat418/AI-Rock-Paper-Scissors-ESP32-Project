# ESP32 Project: Rock-Paper-Scissors Live CNN Inference
**ENMGT 5400 - AI for Engineering Managers**

## Executive Summary
This project serves as a proof-of-concept for a new vision-based gaming device (similar to *Simon* or *Bop It*). The primary objective was to validate the **Xiao ESP32S3 Sense** as a viable, low-cost SOC for edge-AI image classification. 

We successfully developed a 4-class Convolutional Neural Network (CNN) capable of recognizing Rock, Paper, Scissors, and an empty background ("None") in real-time. The system achieves **~87% live inference accuracy**, successfully validating the chipset's capability to stream video and handle real-time computer vision tasks for future product development.

---

## 1. System Architecture & Engineering Design
During development, we identified that the required custom C-compiled neural network module (`emlearn_cnn_fp32`) was absent from the baseline testing firmware. To ensure project delivery and maintain a high framerate, we pivoted to a **Real-Time Streaming Architecture**.

* **The Edge Node (ESP32S3):** Configured as a high-speed Wi-Fi IP Camera. It captures raw image data and streams MJPEG frames continuously over a local Access Point network.
* **The Host Node (Local Machine):** Intercepts the video stream. To prevent OpenCV buffer lag (which causes the AI to predict based on delayed frames), we implemented a multi-threaded background reader that constantly flushes the buffer, ensuring the AI only processes the absolute newest frame.
* **Image Preprocessing:** The live 96x96 color stream is instantly converted to grayscale, downsampled to a 32x32 matrix, and passed through a histogram equalization filter to boost contrast before being fed into the neural network.

---

## 2. Dataset Engineering
Clean data is the most critical component of this classifier. A custom dataset was built utilizing the ESP32 camera to ensure the model trained on the exact sensor characteristics it would see in production.

* **Volume:** Collected well over the required 300 images per class.
* **Overcoming Domain Shift:** Initial testing revealed the CNN was "cheating" by memorizing the background and predicting "Paper" for blank walls. We solved this by introducing a 4th class (**`none`**) containing ~100 images of the room with no hand present. 
* **Data Quality:** Images were captured at a consistent distance while varying the background environment to force the CNN to learn the geometry of the hand rather than the ambient lighting.

---

## 3. The CNN Architecture (TinyML Optimized)
The model was built using TensorFlow/Keras. Because this architecture is ultimately intended for a microcontroller with severe memory constraints, we bypassed standard `MaxPooling2D` layers (which are often unsupported by ultra-lightweight frameworks like TinyMaix) and utilized **Strided Convolutions** to compress the spatial dimensions.

**Layer Breakdown:**
1. **Input Layer (`32x32x1`):** Grayscale image input, normalized to values between 0.0 and 1.0.
2. **Conv2D (`8 filters, 3x3, strides=2,2`):** Extracts basic edges and shapes. The stride of 2 aggressively downsamples the image size by half without requiring a dedicated pooling layer.
3. **Conv2D (`16 filters, 3x3, strides=2,2`):** Extracts more complex features (e.g., the V-shape gap for "Scissors"). Downsamples again.
4. **Flatten:** Unrolls the 2D feature maps into a 1D vector.
5. **Dense (`32 nodes, ReLU`):** A fully connected layer that learns the complex, non-linear relationships between the extracted hand features.
6. **Dense Output (`4 nodes, Softmax`):** Outputs the final probability distribution across the 4 classes (`none`, `paper`, `rock`, `scissors`).

*Total Model Size after conversion to TFLite/TMDL: ~103 KB.*

---

## 4. Performance & Future Improvements
The current system achieves **~87% accuracy** during live, real-time video streaming, which is highly responsive and robust to minor lighting changes thanks to our histogram equalization preprocessing.

**Pathways for Improvement:**
1. **True Edge Deployment:** Once the updated `.bin` firmware containing the compiled C-modules is available, the `.tmdl` model can be pushed directly to the ESP32's flash memory. Moving inference to the edge will eliminate Wi-Fi latency entirely.
2. **Data Augmentation:** While our custom dataset is robust, utilizing Keras's `ImageDataGenerator` to artificially rotate, zoom, and shift the training images would make the model completely immune to camera angle variations.
3. **Advanced Filtering:** Implementing a Sobel Edge Detection filter on the ESP32 before transmission could further reduce the bandwidth required to send frames to the host machine.

---

## 5. Citations & Acknowledgments
* **Hardware & Networking:** Base ESP32 camera configuration, Wi-Fi access point setup, and initial MicroPython `streaming_server` scripts were provided by the ENMGT 5400 course materials and Seeed Studio documentation.
* **AI Assistance:** System debugging, OpenCV multi-threading implementation (for the lag-proof video buffer), and TinyML optimization strategies (strided convolutions vs. max pooling) were developed with the assistance of Google Gemini.
