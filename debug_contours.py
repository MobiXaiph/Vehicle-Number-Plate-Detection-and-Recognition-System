import cv2
import numpy as np
import imutils

def debug_contours():
    image = cv2.imread('car_sample.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200) 
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    
    print(f"Total contours: {len(contours)}")
    
    plate_x, plate_y, plate_w, plate_h = 244, 582, 126, 57
    
    found_match = False
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        
        # Check for overlap/proximity
        if (abs(x - plate_x) < 50 and abs(y - plate_y) < 50):
            print(f"Match found! Contour {i}: x={x}, y={y}, w={w}, h={h}, Area={cv2.contourArea(c)}")
            found_match = True
            
    if not found_match:
        print("No contour found near the plate location.")

if __name__ == "__main__":
    debug_contours()
