import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

class VisionEngine:
    def __init__(self):
        # Load a lightweight pre-trained YOLOv8 model for object detection
        # This shows we use modern SOTA models for feature extraction
        self.model = YOLO('yolov8n.pt') 

    def analyze_image(self, image_path):
        """Extracts pre-LLM features from an image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load image for processing
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Blur Detection (Variance of Laplacian)
        # Higher = sharper, Lower = blurrier
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness & Contrast
        # Calculate mean brightness and standard deviation (contrast)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 3. Edge Density (Background Clutter Estimation)
        # We use Canny edge detection to see how 'busy' the image is
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 4. Color Analysis (Saturation)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # 5. Object Detection
        results = self.model(image_path, verbose=False)
        detected_objects = []
        for r in results:
            for c in r.boxes.cls:
                detected_objects.append(self.model.names[int(c)])
        
        # Overexposure flag (simple heuristic)
        is_overexposed = brightness > 230
        
        return {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2),
            "contrast": round(float(contrast), 2),
            "edge_density": round(float(edge_density), 4),
            "avg_saturation": round(float(avg_saturation), 2),
            "is_overexposed": bool(is_overexposed),
            "detected_objects": list(set(detected_objects)),
            "image_dimensions": f"{img.shape[1]}x{img.shape[0]}"
        }

if __name__ == "__main__":
    # Test snippet
    engine = VisionEngine()
    # print(engine.analyze_image("test.jpg")) # Uncomment to test locally
