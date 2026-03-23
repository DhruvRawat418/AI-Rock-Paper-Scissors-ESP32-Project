import cv2
import urllib.request
import numpy as np

# Make sure this matches Thonny!
url = 'http://192.168.1.229:8080/xiao/Hi-Xiao-Ling'

print("Connecting to ESP32 stream...")

try:
    stream = urllib.request.urlopen(url)
    print("Connected! Buffering video...")
    bytes_data = b''

    while True:
        # Read a larger chunk of video data
        bytes_data += stream.read(8192)

        # \xff\xd8 is the Start of a JPEG, \xff\xd9 is the End
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            # Check if the Start comes BEFORE the End
            if a < b:
                jpg = bytes_data[a:b + 2]
                bytes_data = bytes_data[b + 2:]

                # Double check the image isn't completely empty
                if len(jpg) > 0:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is not None:
                        cv2.imshow('XIAO Live Stream', frame)
            else:
                # If the End came first, throw away the broken half-frame
                bytes_data = bytes_data[a:]

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"Error connecting to stream: {e}")

cv2.destroyAllWindows()