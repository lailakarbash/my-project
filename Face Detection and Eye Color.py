import cv2
import numpy as np

# Function to determine eye color based on HSV values
def get_eye_color(hue, value):
    if value < 50:
        return "black"  # If the brightness value is low, the eye is considered black.
    elif hue >= 100 and hue <= 140:
        return "blue"
    elif hue >= 40 and hue <= 80:
        return "green"
    elif hue >= 10 and hue <= 25:
        return "brown"
    elif hue >= 25 and hue <= 35:
        return "Hazel"
    elif hue >= 85 and hue <= 135:
        return "Gray"
    elif hue >= 30 and hue <= 50:
        return "Amber"
    elif hue >= 140 and hue <= 160:
        return "Violet"
    else:
        return "Undefined"  # If the hue does not match any known colors.

def detect_face_and_eyes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade files for detecting faces and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect eyes within each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region to detect eyes within it
        face_region = gray[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region)
        
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Extract the eye region
            eye = image[y + ey:y + ey + eh, x + ex:x + ex + ew]
            
            # Resize the eye region to improve processing speed
            eye_resized = cv2.resize(eye, (50, 50))
            
            # Convert the eye region to HSV color space for color analysis
            hsv_eye = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2HSV)
            
            # Calculate the average hue and value of the eye region
            avg_hue = np.mean(hsv_eye[:, :, 0])  # Average hue value
            avg_value = np.mean(hsv_eye[:, :, 2])  # Average brightness (value)

            # Determine the eye color based on the average hue and value
            eye_color = get_eye_color(avg_hue, avg_value)

            # Display the eye color name on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, eye_color, (x + ex, y + ey - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the image with the labeled eye color
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Call the function with the image path
detect_face_and_eyes('C:/Users/XPRISTO/Desktop/project-2/photo4.jpg')
