import cv2
import pytesseract
from pytesseract import Output

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


def binarize_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)




# Apply threshold to get a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary_image


def segment_image(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def draw_contours(binary_image, contours):
    # Draw contours on the image
    image_with_contours = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    return image_with_contours


def ocr_image(image):
    # Perform OCR on the image
    text = pytesseract.image_to_string(image, lang='eng')

    return text


def process_image(image_path):
    binary_image = binarize_image(image_path)
    contours = segment_image(binary_image)
    image_with_contours = draw_contours(binary_image, contours)

    # Optionally save or display the image with contours
    #cv2.imwrite('image_with_contours.png', image_with_contours)
    #cv2.imshow('Contours', image_with_contours)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Perform OCR
    text = ocr_image(binary_image)

    return text


if __name__ == '__main__':
    image_path = "/assets/ocrimage.jpg"  # Update this with the path to your image
    text = process_image(image_path)
    print('OCR Text:', text)
