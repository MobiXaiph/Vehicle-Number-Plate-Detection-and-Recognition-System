# Vehicle Number Plate Detection and Recognition System

This project is an automated **Vehicle Number Plate Detection and Recognition System** built using Python, OpenCV, and EasyOCR. It aims to locate vehicle number plates in an image, extract them, and utilize Optical Character Recognition (OCR) to read the characters on the plates accurately.

## 🚀 How It Works (Methodology)

The pipeline is split into two primary stages: **Detection** and **Recognition**.

### 1. Image Preprocessing & Detection
First, the input image goes through a series of transformations to isolate the contours of the number plate:
- **Grayscaling:** The image is converted to grayscale to reduce dimensionality and computational complexity.
- **Bilateral Filtering:** We use a bilateral filter (`cv2.bilateralFilter`) to sharply remove noise while keeping the edges intact—a very important detail when identifying plates.
- **Edge Detection:** Canny Edge Detection (`cv2.Canny`) is applied to find distinct outlines. 
- **Morphological Transformations:** Dilation is used (`cv2.dilate`) to bridge any gaps in the structural components and connect fragmented edges.
- **Contour Filtering:** The algorithm finds closed contours in the edged image. It sorts the largest contours and tests them based on generic rectangular shapes.
- **Aspect Ratio Filtering:** The system further refines candidates using mathematical heuristic—checking if the aspect ratio falls between $2.0$ and $6.0$ (typical aspect ratio for number plates globally).

### 2. Plate Extraction & OCR
Once plate-like regions are found:
- **Perspective Transform:** To account for skewed angles, the image is warped using a Four-Point Perspective Transform (`imutils.perspective.four_point_transform`) into a top-down view. 
- **Axis-Aligned Cropping:** As a fallback mechanism, an axis-aligned bounding box of the cropped image is also generated.
- **Optical Character Recognition (OCR):** The system passes the cropped (both warped and axis-aligned) images to **EasyOCR**. EasyOCR extracts the actual text present within the candidate region.
- **Output Validation:** The program scores the output based on confidence and displays the final matched text and accuracy confidence score overlayed directly onto the vehicle image.

---

## 🛠 Libraries & Technologies Used

- **[Python](https://www.python.org/):** Core programming language used for the tool logic.
- **[OpenCV (`cv2`)](https://opencv.org/):** Used heavily for image manipulation, grayscaling, morphological transformations, and contour analysis.
- **[NumPy (`numpy`)](https://numpy.org/):** For efficient high-level math and multidimensional array/matrix calculations (resolving geometric points).
- **[Imutils (`imutils`)](https://github.com/PyImageSearch/imutils):** A specialized series of convenience functions built to support OpenCV operations like easy perspective skewing/transforms, and simple sorting of grabbed image contours.
- **[EasyOCR (`easyocr`)](https://github.com/JaidedAI/EasyOCR):** The robust OCR engine capable of detecting text in an image without mandating a heavy deep-learning model loaded natively.
- **[Matplotlib (`matplotlib`)](https://matplotlib.org/):** Used conditionally to plot visual breakdowns of intermediate statuses (showing original, edged, cropped representations) back-to-back.

---

## 🤔 How to Setup and Run

### 1. Requirements Installation
It is recommended to run this project in a virtual environment. Install dependencies using:

```bash
pip install opencv-python numpy imutils easyocr matplotlib
```

### 2. Running the System
You can pass your image via the command-line interface. For example:

```bash
python main.py --image path/to/vehicle/image.jpg
```
*(If no argument is passed, it falls back to a default sample image)*

### 3. Output
The system will:
1. Print diagnostic logs outlining the found candidate contours, coordinates, and recognition confidence scores to the terminal.
2. Render a comparison visualization grid breaking down the image morphing process.
3. Display a final pop-up highlighting the specific vehicle plate bounded natively, complete with overlaid recognized text!
