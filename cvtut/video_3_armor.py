import os
import cv2 as cv

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the video path using a relative path
video_path = os.path.join(script_dir, "roboarm.mp4")


video = cv.VideoCapture(video_path)

# Check if the video source is opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the total number of frames in the video
total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Calculate the start frame for the second half
start_frame = 2 * total_frames // 3

# Set the video to the start frame
video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

ret = True
while ret:
    ret, frame = video.read()
    if ret:
        # Convert the frame to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Apply binary thresholding
        _, binary_frame = cv.threshold(gray_frame, 128, 255, cv.THRESH_BINARY)
        
        # Get the dimensions of the binary frame
        height, width = binary_frame.shape
        
        # Define the region of interest (right bottom corner)
        crop_frame = binary_frame[int(height * 0.7): int(height * 0.9), int(2 * width / 3):int(6 * width / 7)]
        
        # Display the cropped binary frame
        cv.imshow('frame', crop_frame)
        

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    

# Release the video capture object and close all OpenCV windows
video.release()
cv.destroyAllWindows()