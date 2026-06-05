# Face-Detection-with-Haar-Cascades

## Aim
To write a Python program using OpenCV to perform the following image manipulations:

i) Extract ROI from an image.

ii) Perform face detection using Haar Cascades in static images.

iii) Perform eye detection in images.

iv) Perform face detection with label in real-time video from webcam.

## Software Required
-Anaconda - Python 3.7 or above
-OpenCV library (opencv-python)
-Matplotlib library (matplotlib)
-Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

I) Load and Display Images

Step 1: Import necessary packages: numpy, cv2, matplotlib.pyplot

Step 2: Load grayscale images using cv2.imread() with flag 0

Step 3: Display images using plt.imshow() with cmap='gray'


II) Load Haar Cascade Classifiers

Step 1: Load face and eye cascade XML files

III) Perform Face Detection in Images

Step 1: Define a function detect_face() that copies the input image

Step 2: Use face_cascade.detectMultiScale() to detect faces

Step 3: Draw white rectangles around detected faces with thickness 10

Step 4: Return the processed image with rectangles


IV) Perform Eye Detection in Images

Step 1: Define a function detect_eyes() that copies the input image

Step 2: Use eye_cascade.detectMultiScale() to detect eyes

Step 3: Draw white rectangles around detected eyes with thickness 10

Step 4: Return the processed image with rectangles


V) Display Detection Results on Images

Step 1: Call detect_face() or detect_eyes() on loaded images

Step 2: Use plt.imshow() with cmap='gray' to display images with detected regions highlighted


VI) Perform Face Detection on Real-Time Webcam Video

Step 1: Capture video from webcam using cv2.VideoCapture(0)

Step 2: Loop to continuously read frames from webcam

Step 3: Apply detect_face() function on each frame

Step 4: Display the video frame with rectangles around detected faces

Step 5: Exit loop and close windows when ESC key (key code 27) is pressed

Step 6: Release video capture and destroy all OpenCV windows

## Program
# Developed by: KATHI HASINI
# Register Number: 212224240074
```
# Developed by: Kathi Hasini
# Register Number: 212224240074

import cv2
import matplotlib.pyplot as plt

# LOAD HAAR CASCADES
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye.xml')

# DETECTION FUNCTIONS
def detect_face(image):
    img_copy = image.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), 3)
    return img_copy

def detect_eyes(image):
    img_copy = image.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_img, 1.1, 3)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), 2)
    return img_copy

# IMAGE LIST
single_images = ["img-01.jpg", "img-02.webp"]
group_image   = "img-03.webp"

# PROCESS SINGLE IMAGES SIDE-BY-SIDE
for image_name in single_images:
    print(f"\nProcessing: {image_name}")

    img = cv2.imread(image_name)
    face_img = detect_face(img)
    eye_img  = detect_eyes(img)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Faces Detected")
    plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Eyes Detected")
    plt.imshow(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# PROCESS GROUP IMAGE SEPARATELY (FOR CLARITY)
print("\nProcessing Group Image Separately for Clear View...")

img_group = cv2.imread(group_image)
face_group = detect_face(img_group)
eye_group  = detect_eyes(img_group)

plt.figure(figsize=(14, 7))
plt.title("Group Image - Original")
plt.imshow(cv2.cvtColor(img_group, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(14, 7))
plt.title("Group Image - Faces Detected")
plt.imshow(cv2.cvtColor(face_group, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(14, 7))
plt.title("Group Image - Eyes Detected")
plt.imshow(cv2.cvtColor(eye_group, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print("\nAll images processed successfully!")

# REAL-TIME FACE DETECTION USING WEBCAM
print("\nStarting Real-Time Face Detection...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()
else:
    print("Webcam connected successfully. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture frame!")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    cv2.imshow("Real-Time Face Detection (ESC to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\nReal-Time Face Detection Ended.")

```
## Output
<img width="851" height="226" alt="image" src="https://github.com/user-attachments/assets/040c5eea-1745-49f0-bb34-b6a4469fd04e" />

<img width="850" height="352" alt="image" src="https://github.com/user-attachments/assets/ed36d0cc-e40e-40ea-92b9-cb135fb099bd" />

<img width="1233" height="707" alt="image" src="https://github.com/user-attachments/assets/a5a771b9-1bc5-495d-8a34-ce6484cc4a33" />

<img width="1178" height="711" alt="image" src="https://github.com/user-attachments/assets/bc7af2ba-4f05-4c5d-955c-ee14480d71cc" />

<img width="1221" height="712" alt="image" src="https://github.com/user-attachments/assets/b3cd8fa2-3edb-4b68-b6a0-8f65b98731e9" />

## Real-time Face Detection:
<img width="791" height="636" alt="Screenshot 2026-06-05 211512" src="https://github.com/user-attachments/assets/1e2bcc9a-a11d-4e36-b713-065e3521740a" />

## Result
Thus, face and eye detection were successfully implemented using Haar Cascades in Python. The program detected faces and eyes in static images and performed labeled real-time face detection from webcam feed using OpenCV
