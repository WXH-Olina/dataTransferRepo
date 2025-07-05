import cv2 as cv
import numpy as np

# Create an empty 1280 x 720 image with a gradient background
img = np.zeros((720, 1280, 3), dtype=np.uint8)
for i in range(img.shape[0]):
    img[i, :] = [int(200 * (i / img.shape[0])), int(200 * (i / img.shape[0])), int(200 * (i / img.shape[0]))]

# Draw a rounded rectangle at the center of the image
center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
radius, thickness = 30, -1

# Draw shadow
cv.rectangle(img, (center_x - 505, center_y - 55), (center_x + 505, center_y + 55), (255, 255, 255), thickness)
cv.rectangle(img, (center_x - 500, center_y - 50), (center_x + 500, center_y + 50), (0, 0, 0), thickness)

# Write the text
font = cv.FONT_HERSHEY_COMPLEX
text = "Congrats! You have successfully installed OpenCV."
text_size = cv.getTextSize(text, font, 1, 2)[0]
cv.putText(img, text, (center_x - text_size[0] // 2, center_y + text_size[1] // 2 - 5), font, 1, (255, 255, 255), 2)

# Display the image
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the image as "Image1.png" in the current directory
# cv.imwrite("Image1.png", img)