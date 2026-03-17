import easyocr
import cv2

def test_ocr():
    reader = easyocr.Reader(['en'], gpu=False)
    image = cv2.imread('plate_crop_simple.png')
    results = reader.readtext(image)
    
    print(f"Found {len(results)} text blocks")
    for (bbox, text, prob) in results:
        print(f"Text: '{text}', Prob: {prob:.2f}, Box: {bbox}")

if __name__ == "__main__":
    test_ocr()
