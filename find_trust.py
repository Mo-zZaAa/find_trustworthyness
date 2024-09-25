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
widths = []
heights = []
fWHRs = []
eyebrows = []
chin_angles = []
philtrums = []
face_shapes = []
trustworthiness_indices = []

start_time = time.time()
measurement_duration = 5  # 3초로 측정 시간 설정

# Function to calculate midpoint
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are (x, y) coordinates
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

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
        
        # Calculate face width (P2 to P16)
        P2 = shape[1]   # Point 2
        P16 = shape[15]  # Point 16
        face_width = np.linalg.norm(P2 - P16)
        
        # Calculate face height (Midpoint(P22, P23) to P52)
        P22 = shape[21]  # Point 22
        P23 = shape[22]  # Point 23
        midpoint_P22_P23 = midpoint(P22, P23)  # Midpoint of P22 and P23
        P52 = shape[51]  # Point 52 (between the nose and lips)
        face_height = np.linalg.norm(np.array(midpoint_P22_P23) - P52)
        
        # Calculate fWHR
        fWHR = face_width / face_height if face_height > 0 else 0
        fWHRs.append(fWHR)
        widths.append(face_width)
        heights.append(face_height)

        # Calculate inner eyebrow angle
        left_eyebrow_angle = calculate_angle(shape[22], shape[21], shape[20])
        right_eyebrow_angle = calculate_angle(shape[23], shape[24], shape[25])
        avg_eyebrow_angle = (left_eyebrow_angle + right_eyebrow_angle) / 2
        eyebrows.append(avg_eyebrow_angle)

        # Calculate chin angle
        chin_angle = calculate_angle(shape[8], shape[9], shape[7])  # Using chin points
        chin_angles.append(chin_angle)

        # Calculate philtrum length (distance between the nose and upper lip)
        philtrum_length = np.linalg.norm(shape[33] - shape[51])
        philtrums.append(philtrum_length)

        # Calculate face roundness (aspect ratio of face)
        face_shape_ratio = face_width / face_height
        face_shapes.append(face_shape_ratio)

        # Draw landmarks and lines
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.line(image, tuple(P2), tuple(P16), (255, 0, 0), 1)  # Line for width
        cv2.line(image, midpoint_P22_P23, tuple(P52), (0, 0, 255), 1)  # Line for height
    
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

# Calculate final average facial metrics
final_width = np.mean(widths) if widths else 0
final_height = np.mean(heights) if heights else 0
final_ratio = final_width / final_height if final_height > 0 else 0

# 표준화 함수
def standardize(values):
    return (values - np.mean(values)) / np.std(values)

# 표준화된 값 계산
eyebrows_std = standardize(eyebrows) * -1
face_shape_std = standardize(face_shapes)
chin_angle_std = standardize(chin_angles)
philtrum_std = standardize(philtrums) * -1
fWHR_std = standardize(fWHRs) * -1  # fWHR은 신뢰도가 낮아지는 방향으로 처리

# 신뢰도 지수 계산
trustworthiness_index = (eyebrows_std + face_shape_std + chin_angle_std + philtrum_std + fWHR_std) / 5

# 평균 신뢰도 점수 계산
average_trustworthiness_index = np.mean(trustworthiness_index)

# 0에서 1 사이로 정규화한 후 0에서 100으로 변환
min_value = min(trustworthiness_index)
max_value = max(trustworthiness_index)
normalized_index = [(score - min_value) / (max_value - min_value) * 100 for score in trustworthiness_index]

# Prepare data for Excel
data = {
    'Width': widths,
    'Height': heights,
    'Ratio': [w / h if h > 0 else 0 for w, h in zip(widths, heights)],
    'Eyebrow_Angle': eyebrows,
    'Chin_Angle': chin_angles,
    'Philtrum_Length': philtrums,
    'Face_Shape_Ratio': face_shapes,
    'Trustworthiness_Index': trustworthiness_index,
    'Trustworthiness_Score': normalized_index
}

df = pd.DataFrame(data)

# Save the results to an Excel file
output_file = r"C:\Users\세은\OneDrive\바탕 화면\someones\facial-landmarks-recognition\fwhr_trustworthiness_results.xlsx"
df.to_excel(output_file, index=False)

# Print the final average ratio and trustworthiness index
print(f"Your final average facial width-to-height ratio is {final_ratio:.2f}")
print(f"Average Trustworthiness Index: {average_trustworthiness_index:.2f}")
print(f"Results saved to {output_file}")
