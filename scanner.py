# 1. INSTALL DEPENDENCIES
!apt-get install -y poppler-utils
!pip install pdf2image imutils

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from pdf2image import convert_from_path
from google.colab import files
from PIL import Image

def scan_document_ultra(image):
    """
    Complete Pipeline: 
    Preprocessing -> Edge Detection -> Corner Detection -> 
    Perspective Warp -> B&W Thresholding -> Denoising -> Auto-Crop
    """
    # Store ratio for high-res warping
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image_resized = imutils.resize(image, height=500)

    # --- STEP 1: PREPROCESSING ---
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Dilation to bridge gaps in document boundaries
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    # --- STEP 2: CORNER DETECTION ---
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # --- STEP 3: PERSPECTIVE WARP ---
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        warped = orig

    # --- STEP 4: B&W EFFECT & DENOISING ---
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Adaptive Thresholding handles uneven lighting
    scanned = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 15
    )
    
    # Morphological Opening to remove small noise/dots
    noise_kernel = np.ones((2,2), np.uint8)
    scanned = cv2.morphologyEx(scanned, cv2.MORPH_OPEN, noise_kernel)

    # --- STEP 5: SMART CONTENT CROP ---
    inv = cv2.bitwise_not(scanned)
    points = np.argwhere(inv > 0)
    
    if len(points) > 0:
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        
        # Add padding for a professional margin
        pad = 50
        h, w = scanned.shape
        scanned = scanned[max(0, y_min-pad):min(h, y_max+pad), 
                          max(0, x_min-pad):min(w, x_max+pad)]
        
        # Add a final uniform white border
        scanned = cv2.copyMakeBorder(scanned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    return scanned

# --- EXECUTION BLOCK ---
print("🚀 Upload your documents (Image or PDF):")
uploaded = files.upload()
scanned_pages = []

for filename in uploaded.keys():
    if filename.lower().endswith(".pdf"):
        pages = convert_from_path(filename)
        images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
    else:
        images = [cv2.imread(filename)]

    for i, img in enumerate(images):
        if img is None: continue
        
        result = scan_document_ultra(img)
        scanned_pages.append(Image.fromarray(result))
        
        # Plotting results
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        plt.title("Original Capture")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Ultra Scanned & Cropped")
        plt.imshow(result, cmap="gray")
        plt.axis("off")
        plt.show()

# Save as consolidated PDF
if scanned_pages:
    output_name = "Digital_Scan_Output.pdf"
    scanned_pages[0].save(output_name, save_all=True, append_images=scanned_pages[1:])
    print(f"✅ Success! Download '{output_name}' from the sidebar.")
