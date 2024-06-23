import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the image
img = cv2.imread("C:/Dataset FDS/temp/inscriptionimages/testimage.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to binarize the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Use Tesseract OCR to detect text regions
text_regions = pytesseract.image_to_boxes(img, lang='tam', config='--psm 11')

# Parse the text regions to identify keypoints
img_height, img_width = img.shape[:2]
keypoints = []
for region in text_regions.splitlines():
    region = region.split()
    x1, y1 = int(region[1]), int(region[2])
    x2, y2 = int(region[3]), int(region[4])
    # Adjust y-coordinates because Tesseract's origin is bottom-left
    y1 = img_height - y1
    y2 = img_height - y2
    # Add keypoints for the four corners of the bounding box
    keypoints.append((x1, y1))
    keypoints.append((x2, y1))
    keypoints.append((x1, y2))
    keypoints.append((x2, y2))
boxes = []

# Draw keypoints on the original image
img_with_keypoints = img.copy()
for point in keypoints:
    cv2.circle(img_with_keypoints, point, 3, (0, 255, 0), -1)
# Resize the image for display
max_height, max_width = 800, 800  # Max height and width for the display window
aspect_ratio = img_with_keypoints.shape[1] / img_with_keypoints.shape[0]

if img_with_keypoints.shape[0] > max_height or img_with_keypoints.shape[1] > max_width:
    if aspect_ratio > 1:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)
    img_with_keypoints = cv2.resize(img_with_keypoints, (new_width, new_height))

# Display the image with keypoints
cv2.imshow("Image with Keypoints", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()


