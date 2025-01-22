import cv2
import numpy as np

# Function to determine the eye color based on Hue and Value in the HSV color space
def get_eye_color(hue, value):
    if value < 50:
        return "black"  # If the brightness is low, the eye is considered black.
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
        return "Undefined"  # If the color doesn't match known color ranges

# Function to determine the skin tone based on Hue, Saturation, and Value in the HSV color space
def get_skin_tone(hue, saturation, value):
    if saturation < 40:
        return "Pale"  # If saturation is low, the skin is considered pale
    elif hue >= 0 and hue <= 25:
        return "Light skin"
    elif hue >= 25 and hue <= 40:
        return "Medium light skin"
    elif hue >= 40 and hue <= 60:
        return "Medium skin"
    elif hue >= 60 and hue <= 90:
        return "Medium dark skin"
    elif hue >= 90 and hue <= 140:
        return "Dark skin"
    else:
        return "Unknown skin tone"  # If the color doesn't match known skin tones

# Function to detect face and eyes, then classify their colors and skin tone
def detect_face_and_eyes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or failed to load.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade files for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Failed to load Haar Cascade file for face detection.")
        return

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        print("Failed to load Haar Cascade file for eye detection.")
        return

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each detected face, detect eyes and classify their color
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face
        face_region = image[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region)
        
        # Ensure only two eyes are detected
        eyes_detected = 0  # Counter for detected eyes
        for (ex, ey, ew, eh) in eyes:
            if eyes_detected < 2:  # If fewer than two eyes are detected
                cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # Draw a rectangle around the eye
                eye = image[y + ey:y + ey + eh, x + ex:x + ex + ew]  # Extract the eye region
                eye_resized = cv2.resize(eye, (50, 50))  # Resize the eye image for better processing speed
                hsv_eye = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2HSV)  # Convert the eye image to HSV
                avg_hue = np.mean(hsv_eye[:, :, 0])  # Get the average Hue value
                avg_value = np.mean(hsv_eye[:, :, 2])  # Get the average Value (brightness)
                eye_color = get_eye_color(avg_hue, avg_value)  # Determine the eye color

                # Display the eye color on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, eye_color, (x + ex, y + ey - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                eyes_detected += 1  # Increment the eye counter

        # Extract the skin region (area below the face, roughly the neck and lower face)
        skin_region = image[y + int(h/4):y + h, x:x + w]
        hsv_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)  # Convert the skin region to HSV
        avg_hue_skin = np.mean(hsv_skin[:, :, 0])  # Get the average Hue of the skin
        avg_saturation_skin = np.mean(hsv_skin[:, :, 1])  # Get the average Saturation of the skin
        avg_value_skin = np.mean(hsv_skin[:, :, 2])  # Get the average Value of the skin
        skin_tone = get_skin_tone(avg_hue_skin, avg_saturation_skin, avg_value_skin)  # Determine the skin tone

        # Display the skin tone on the image
        cv2.putText(image, skin_tone, (x, y - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the image with face, eye colors, and skin tone
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path
detect_face_and_eyes('C:/Users/XPRISTO/Desktop/advanced programing/projekt/pp2.png')
