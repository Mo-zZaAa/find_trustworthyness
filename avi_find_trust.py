import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
import os
from multiprocessing import Pool

# Initialize dlib's face detector and facial landmark predictor
predictor_path = r"C:\facelandmark\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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

# Function to standardize values
def standardize(values):
    return (values - np.mean(values)) / np.std(values) if len(values) > 0 else values

# Function to process a single video file
def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Starting processing for {video_name}")

    cap = cv2.VideoCapture(video_path)

    # Variables for storing results
    frame_number = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or cannot read the file: {video_name}")
            break

        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = detector(gray, 1)

        # Process each detected face separately
        for i, rect in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            fWHR, width, height = calculate_fWHR(shape)
            eyebrow_angle = np.arctan2(shape[19][1] - shape[24][1], shape[24][0] - shape[19][0]) * 180 / np.pi
            chin_angle = np.arctan2(shape[8][1] - shape[57][1], shape[57][0] - shape[8][0]) * 180 / np.pi
            philtrum_length = np.linalg.norm(shape[33] - shape[51])
            face_shape_ratio = width / height

            results.append({
                'Frame_Number': frame_number,
                'Face_ID': i,  # Face ID for each detected face
                'Width': width,
                'Height': height,
                'fWHR': fWHR,
                'Eyebrow_Angle': eyebrow_angle,
                'Chin_Angle': chin_angle,
                'Philtrum_Length': philtrum_length,
                'Face_Shape_Ratio': face_shape_ratio
            })

    cap.release()
    cv2.destroyAllWindows()

    # Check if any faces were processed
    if not results:
        print(f"No faces detected in {video_name}. Skipping this video.")
        return

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Standardize values and calculate trustworthiness index for each face
    df['Eyebrow_Angle_Std'] = standardize(df['Eyebrow_Angle']) * -1
    df['Face_Shape_Std'] = standardize(df['Face_Shape_Ratio'])
    df['Chin_Angle_Std'] = standardize(df['Chin_Angle'])
    df['Philtrum_Length_Std'] = standardize(df['Philtrum_Length']) * -1
    df['fWHR_Std'] = standardize(df['fWHR']) * -1
    df['Trustworthiness_Index'] = (df['Eyebrow_Angle_Std'] + df['Face_Shape_Std'] +
                                   df['Chin_Angle_Std'] + df['Philtrum_Length_Std'] +
                                   df['fWHR_Std']) / 5

    # Normalize Trustworthiness Index
    min_value, max_value = df['Trustworthiness_Index'].min(), df['Trustworthiness_Index'].max()
    df['Trustworthiness_Score'] = ((df['Trustworthiness_Index'] - min_value) /
                                   (max_value - min_value) * 100).fillna(0)

    # Save results to Excel
    output_file = os.path.join(r"C:\Users\세은\OneDrive\바탕 화면", f"{video_name}_trustworthiness_results.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Finished processing for {video_name}. Results saved to {output_file}")

# Main function to process the specified video files with multiprocessing
def process_all_videos(video_paths):
    with Pool() as pool:
        pool.map(process_video, video_paths)

if __name__ == '__main__':
    video_paths = [
        r"C:\Users\세은\Downloads\2203_Berkshire Hathaway_Warren Buffett_120208.avi",
        r"C:\Users\세은\Downloads\2330_Berkshire Hathaway_Warren Buffet_030308.avi",
        r"C:\Users\세은\Downloads\2345_Assured Guaranty_Dominic Frederico_040308.avi"
    ]

    process_all_videos(video_paths)
