# Vehicle Plate Recognition Project Report

## 1. Project Overview
The **Vehicle Plate Recognition System** is a computer vision application designed to automatically detect and recognize license plates from images of vehicles. It utilizes image processing techniques for plate localization and Optical Character Recognition (OCR) for reading the alphanumeric characters on the plate.

## 2. Objective
The primary objective is to build a robust system that can:
1.  Input an image of a vehicle.
2.  Locate the license plate within the image.
3.  Extract and correct the perspective of the plate.
4.  Read and output the text from the license plate.

## 3. Methodology & Technologies
The project is implemented in **Python** using the following key libraries:
*   **OpenCV (`cv2`)**: For all image processing tasks including grayscale conversion, edge detection, contour finding, and perspective transformation.
*   **EasyOCR**: A deep learning-based OCR library used to read text from the detected plate regions.
*   **Imutils**: Helper functions for image processing.
*   **Matplotlib**: For visualizing the intermediate steps and final results.

## 4. System Workflow

The processing pipeline consists of five main stages:

1.  **Image Loading**: The system reads the input image provided by the user.
2.  **Preprocessing**:
    *   **Grayscale Conversion**: Simplifies the image data.
    *   **Bilateral Filter**: Reduces noise while preserving edges, which is critical for accurate detection.
    *   **Canny Edge Detection**: Identifies structural edges in the image.
3.  **Plate Localization**:
    *   **Contour Detection**: Finds closed shapes in the edge map.
    *   **Candidate Filtering**: Filters contours based on area and aspect ratio (rectangular shape typical of license plates).
    *   **Perspective Transform**: warps the detected region to a top-down "bird's eye" view for better OCR accuracy.
4.  **Character Recognition (OCR)**:
    *   The extracted plate image is passed to the EasyOCR engine.
    *   The engine extracts text and confidence scores.
5.  **Result Display**: The system overlays the detected text on the original image and displays the processing stages.

## 5. How to Run & Upload Images

**To run the system with your own image:**

1.  **Upload**: Copy your image file (e.g., `car.jpg`) into the project folder:  
    `c:\Users\Admin\.gemini\antigravity\scratch\vehicle_plate_recognition\`
2.  **Execute**: Open a terminal in that folder and run:
    ```bash
    python main.py -i car.jpg
    ```

---

## 6. Source Code

Below is the complete source code for the project, consolidated for your report.

### File: `main.py`
*Entry point of the application. Handles command-line arguments and orchestrates the flow.*

```python
import cv2
import os
import argparse
from plate_detector import PlateDetector
from ocr_engine import OCREngine
from utils import plot_images, show_result

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load Image
    img = cv2.imread(image_path)
    
    # 2. Initialize Detector and OCR
    detector = PlateDetector()
    ocr = OCREngine()

    # 3. Preprocessing & Detection
    gray, edged = detector.preprocess_image(img)
    candidates = detector.find_plate_contours(edged)
    
    if not candidates:
        print("No number plate contour found.")
        return

    print(f"Testing {len(candidates)} candidates...")
    
    found_text = None
    found_conf = 0.0
    final_warped = None
    final_axis = None

    for i, location in enumerate(candidates):
        print(f"Candidate {i+1}:")
        warped, axis_aligned = detector.extract_plate(img, location)
        
        # Try warped
        text, confidence = ocr.read_text(warped)
        if text == "No Text Found":
            # Try axis-aligned
            text, confidence = ocr.read_text(axis_aligned)
        
        if text != "No Text Found" and confidence > 0.5:
            print(f"MATCH FOUND: {text} ({confidence:.2f})")
            found_text = text
            found_conf = confidence
            final_warped = warped
            final_axis = axis_aligned
            break
        else:
            print(f"  No valid text (got '{text}', conf={confidence:.2f})")

    if found_text:
        print(f"Final Detection: {found_text}")
        # 5. Visualization
        plot_images(
            [img, edged, final_warped, final_axis], 
            ['Original', 'Edges', 'Warped', 'Axis Aligned'],
            figsize=(12, 8)
        )
        show_result(img, found_text, found_conf)
    else:
        print("Could not detect text on any candidate.")

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default='car_sample.png',
        help="path to input vehicle image")
    args = vars(ap.parse_args())

    IMAGE_PATH = args["image"]
    main(IMAGE_PATH)
```

### File: `plate_detector.py`
*Contains the logic for finding and extracting the license plate from the image.*

```python
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

class PlateDetector:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        """
        Converts to grayscale, applies bilateral filter, and finds edges.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bilateral filter removes noise while keeping edges sharp
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        edged = cv2.Canny(bfilter, 30, 200) 
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        return gray, edged

    def find_plate_contours(self, edged):
        """
        Finds contours and filters them to find the number plate.
        """
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        candidates = []
        print(f"Found {len(contours)} contours")
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 3000:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            (x, y), (width, height), angle = rect
            
            # Calculate aspect ratio
            # Width and height can be swapped depending on rotation
            if width > height:
                ar = width / height
            else:
                ar = height / width
            
            print(f"Contour {i}: area={area}, ar={ar:.2f}")

            if 2.0 <= ar <= 6.0:
                candidates.append(box)
        
        return candidates

    def extract_plate(self, image, location):
        """
        Extracts the plate region using perspective transform.
        """
        if location is None:
            return None, None

        # Reshape to 4x2 array of coordinates
        pts = location.reshape(4, 2)
        warped = four_point_transform(image, pts)
        print(f"Extracted plate size: {warped.shape}")
        
        # Axis-aligned crop
        rect = cv2.boundingRect(location)
        x, y, w, h = rect
        
        # Add padding
        padding = 10
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Clip to image bounds
        h_img, w_img = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        print(f"Axis-aligned crop: x={x}, y={y}, w={w}, h={h}")
        axis_aligned = image[y:y+h, x:x+w]
        
        return warped, axis_aligned
```

### File: `ocr_engine.py`
*Handles the Optical Character Recognition using EasyOCR.*

```python
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
```

### File: `utils.py`
*Helper functions for visualization.*

```python
import matplotlib.pyplot as plt
import cv2

def plot_images(images, titles, cmap='gray', figsize=(15, 5)):
    """
    Plots a list of images with titles.
    """
    count = len(images)
    plt.figure(figsize=figsize)
    for i in range(count):
        plt.subplot(1, count, i + 1)
        if len(images[i].shape) == 3:
            # Convert BGR to RGB for display if it's a color image
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('debug_output.png')
    # plt.show()

def show_result(image, text, confidence):
    """
    Displays the final result with the detected text.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected: '{text}' ({confidence:.2f})")
    plt.axis('off')
    plt.savefig('result_output.png')
    # plt.show()
```
