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
