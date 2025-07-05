import os
import cv2 as cv

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the video path using a relative path
video_path = os.path.join(script_dir, "roboarm.mp4")
video = cv.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()


ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv.imshow('frame', frame)
        # Check for 'q' key press
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

video.release()
cv.destroyAllWindows()


