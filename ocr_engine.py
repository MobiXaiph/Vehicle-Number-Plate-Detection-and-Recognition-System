import easyocr
import cv2

class OCREngine:
    def __init__(self, languages=['en']):
        """
        Initializes the EasyOCR reader.
        """
        # gpu=False for compatibility if CUDA is not available, though True is faster
        self.reader = easyocr.Reader(languages, gpu=False) 

    def read_text(self, image_crop):
        """
        Reads text from the cropped image.
        Returns the text and confidence of the best match.
        """
        if image_crop is None:
            return "No Plate Detected", 0.0

        # Upscale if too small
        if image_crop.shape[0] < 64:
            scale = 64 / image_crop.shape[0]
            width = int(image_crop.shape[1] * scale)
            height = 64
            image_crop = cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_CUBIC)
            print(f"Upscaled to: {image_crop.shape}")

        # Check rotation (if taller than wide)
        if image_crop.shape[0] > image_crop.shape[1]:
            print("Image is tall, rotating 90 degrees...")
            image_crop = cv2.rotate(image_crop, cv2.ROTATE_90_CLOCKWISE)

        result = self.reader.readtext(image_crop)
        
        if not result:
            return "No Text Found", 0.0

        # result format: list of (bbox, text, prob)
        # We take the one with highest probability or just the first one if simple
        # Usually for a plate crop, we expect one main text
        
        # Join all detected text blocks
        full_text = []
        confidences = []
        for (_, t, c) in result:
            full_text.append(t)
            confidences.append(c)
        
        text = " ".join(full_text)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return text, confidence
