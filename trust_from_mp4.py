import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
predictor_path = r"C:\facelandmark\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize video capture for an mp4 file
video_path = r"C:\Users\세은\Downloads\your_video_file.mp4"
cap = cv2.VideoCapture(video_path)

# Variables for storing results
frame_number = 0
widths = []
heights = []
fWHRs = []
eyebrows = []
chin_angles = []
philtrums = []
face_shapes = []
trustworthiness_indices = []
valid_frames = []

# Function to calculate midpoint
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

# Function to calculate fWHR
def calculate_fWHR(shape):
    P2 = shape[1]
    P16 = shape[15]
    face_width = np.linalg.norm(P2 - P16)
    P22 = shape[21]
    P23 = shape[22]
    midpoint_P22_P23 = midpoint(P22, P23)
    P52 = shape[51]
    face_height = np.linalg.norm(np.array(midpoint_P22_P23) - P52)
    fWHR = face_width / face_height if face_height > 0 else 0
    return fWHR, face_width, face_height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the file.")
        break

    frame_number += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detector(gray, 1)
    if len(rects) > 0:
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        fWHR, width, height = calculate_fWHR(shape)
        fWHRs.append(fWHR)
        widths.append(width)
        heights.append(height)
        
        eyebrow_angle = np.arctan2(shape[19][1] - shape[24][1], shape[24][0] - shape[19][0]) * 180 / np.pi
        chin_angle = np.arctan2(shape[8][1] - shape[57][1], shape[57][0] - shape[8][0]) * 180 / np.pi
        philtrum_length = np.linalg.norm(shape[33] - shape[51])
        face_shape_ratio = width / height

        eyebrows.append(eyebrow_angle)
        chin_angles.append(chin_angle)
        philtrums.append(philtrum_length)
        face_shapes.append(face_shape_ratio)
        valid_frames.append(frame_number)

cap.release()
cv2.destroyAllWindows()

def standardize(values):
    return (values - np.mean(values)) / np.std(values)

eyebrows_std = standardize(eyebrows) * -1
face_shape_std = standardize(face_shapes)
chin_angle_std = standardize(chin_angles)
philtrum_std = standardize(philtrums) * -1
fWHR_std = standardize(fWHRs) * -1

trustworthiness_index = (eyebrows_std + face_shape_std + chin_angle_std + philtrum_std + fWHR_std) / 5

# 평균 신뢰도 점수 계산
average_trustworthiness_index = np.mean(trustworthiness_index)

# 0에서 1 사이로 정규화한 후 0에서 100으로 변환
min_value = min(trustworthiness_index)
max_value = max(trustworthiness_index)
normalized_index = [(score - min_value) / (max_value - min_value) * 100 for score in trustworthiness_index]

df = pd.DataFrame({
    'Frame_Number': valid_frames,
    'Width': widths,
    'Height': heights,
    'fWHR': fWHRs,
    'Eyebrow_Angle': eyebrows,
    'Chin_Angle': chin_angles,
    'Philtrum_Length': philtrums,
    'Face_Shape_Ratio': face_shapes,
    'Trustworthiness_Index': trustworthiness_index,
    'Trustworthiness_Score': normalized_index
})

output_file = r"C:\Users\세은\OneDrive\바탕 화면\trustworthiness_results_with_average.xlsx"
df.to_excel(output_file, index=False)
print(f"Results saved successfully to {output_file}")
print(f"Average Trustworthiness Index: {average_trustworthiness_index:.2f}")
