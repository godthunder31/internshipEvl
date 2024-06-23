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
text_regions = pytesseract.image_to_boxes(img, lang='eng', config='--psm 11')

# Parse the text regions to identify bounding boxes
img_height, img_width = img.shape[:2]
boxes = []
for region in text_regions.splitlines():
    region = region.split()
    x, y, w, h = int(region[1]), int(region[2]), int(region[3]), int(region[4])
    # Adjust coordinates because Tesseract's origin is bottom-left
    x1 = x
    y1 = img_height - y
    x2 = w
    y2 = img_height - h
    boxes.append((x1, y1, x2, y2))

# Draw bounding boxes on the original image
img_with_boxes = img.copy()
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
