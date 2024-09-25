import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
import time

# Initialize dlib's face detector and facial landmark predictor
predictor_path = r"C:\facelandmark\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for calculating average
num_frames = 0
widths = []
heights = []

start_time = time.time()
measurement_duration = 10  # seconds

while True:
    # Capture frame-by-frame
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate face width and height
        x_coords = shape[:, 0]
        y_coords = shape[:, 1]
        
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        
        # Append to lists for averaging
        widths.append(face_width)
        heights.append(face_height)
        
        # Draw landmarks
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Calculate and display the average width-to-height ratio
    if widths and heights:
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_ratio = avg_width / avg_height if avg_height > 0 else 0
        cv2.putText(image, f'Ratio: {avg_ratio:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the output image
    cv2.imshow("Output", image)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Check if measurement duration has passed
    if time.time() - start_time > measurement_duration:
        break

# Clean up
cv2.destroyAllWindows()
cap.release()

# Calculate the final average ratio
final_width = np.mean(widths) if widths else 0
final_height = np.mean(heights) if heights else 0
final_ratio = final_width / final_height if final_height > 0 else 0

# Print the result
print(f"Your facial width-to-height ratio is {final_ratio:.2f}")

# Save the results to an Excel file
data = {
    'Width': widths,
    'Height': heights,
    'Ratio': [w / h if h > 0 else 0 for w, h in zip(widths, heights)]
}
df = pd.DataFrame(data)
df.to_excel(r"C:\Users\세은\OneDrive\바탕 화면\someones\facial-landmarks-recognition\fwhr_results.xlsx", index=False)
