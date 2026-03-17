import cv2
import os
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

import argparse

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default='car_sample.png',
        help="path to input vehicle image")
    args = vars(ap.parse_args())

    IMAGE_PATH = args["image"]
    main(IMAGE_PATH)
