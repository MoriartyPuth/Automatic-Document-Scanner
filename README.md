# Automatic Document Scanner 📄🚀

A robust, Computer Vision-powered document scanner that transforms skewed smartphone photos into high-contrast, rectified, and professional digital scans. 

## ✨ Key Features
- **Intelligent Edge Detection**: Uses Canny algorithms and Morphological Dilation to capture document boundaries, even against cluttered backgrounds.
- **Perspective Rectification**: Implements `imutils` four-point perspective transformation to correct camera tilt and skew.
- **Adaptive B&W Enhancement**: Utilizes Gaussian adaptive thresholding to eliminate shadows and uneven lighting.
- **Ultra-Cleaning**: Morphological denoising removes "salt and pepper" noise from paper textures.
- **Content-Aware Auto-Crop**: Automatically detects the ink/text area and crops out unnecessary white margins.
- **Multi-Format Support**: Processes both standard image files (JPG/PNG) and PDF documents.

## 🛠️ Technology Stack
- **Python**
- **OpenCV**: Core Image Processing.
- **imutils**: Perspective and resizing convenience.
- **NumPy**: Linear algebra and coordinate sorting.
- **Matplotlib**: Visualization.
- **pdf2image**: PDF rasterization.

## 📐 How it Works (The Pipeline)
1. **Preprocessing**: Grayscale conversion and Gaussian blurring.
2. **Edge Mapping**: Canny Edge Detection + Dilation.
3. **Quadrilateral Localization**: Contour approximation to find exactly 4 vertices.
4. **Warping**: Homography matrix calculation for a top-down view.
5. **Scanning Effect**: Adaptive thresholding and morphological noise reduction.
6. **Smart Cropping**: Pixel-intensity analysis to isolate text content.

## 🚀 Usage
1. Open the project in **Google Colab**.
2. Install dependencies: `pip install imutils pdf2image`.
3. Run the script and upload your image or PDF.
4. Download the consolidated `Digital_Scan_Output.pdf`.

## 📜 References
- Rosebrock, A. (2015). [imutils](https://github.com/PyImageSearch/imutils).
- Canny, J. (1986). A Computational Approach to Edge Detection.
- Douglas-Peucker Algorithm for contour approximation.

---
Developed as a technical demonstration of Computer Vision Image Rectification.
