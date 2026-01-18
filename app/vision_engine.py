import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

class VisionEngine:
    def __init__(self):
        # Load YOLO for basic detection
        self.model = YOLO('yolov8n.pt')
        
        # Load CLIP for accurate product classification
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_available = True
        except Exception as e:
            print(f"CLIP not available: {e}. Falling back to YOLO only.")
            self.clip_available = False

    def _classify_with_clip(self, image_path):
        """Use CLIP to classify the product into e-commerce categories."""
        if not self.clip_available:
            return []
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # E-commerce product categories
            candidate_labels = [
                "athletic footwear", "running shoes", "sneakers",
                "power bank", "portable charger", "electronic device",
                "clothing", "apparel", "t-shirt", "jacket",
                "watch", "smartwatch", "accessory",
                "headphones", "earbuds", "audio device",
                "backpack", "bag", "luggage",
                "camera", "photography equipment",
                "phone case", "mobile accessory",
                "sunglasses", "eyewear"
            ]
            
            inputs = self.clip_processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get top 2 predictions
            top_indices = probs[0].argsort(descending=True)[:2]
            results = []
            for idx in top_indices:
                if probs[0][idx] > 0.15:  # Confidence threshold
                    results.append(candidate_labels[idx])
            
            return results
        except Exception as e:
            print(f"CLIP classification error: {e}")
            return []

    def analyze_image(self, image_path):
        """Extracts pre-LLM features including accurate product classification."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur Detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness & Contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 3. Edge Density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 4. Color Saturation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # 5. Text Detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions_detected = sum(1 for c in contours if cv2.boundingRect(c)[2]/max(cv2.boundingRect(c)[3], 1) > 2)
        
        # 6. YOLO Detection (Fallback)
        results = self.model(image_path, conf=0.15, verbose=False)
        yolo_labels = [self.model.names[int(c)] for r in results for c in r.boxes.cls]
        
        # 7. CLIP Classification (Primary)
        clip_labels = self._classify_with_clip(image_path)
        
        # 8. Saliency
        h_img, w_img = gray.shape
        center_strip = gray[:, int(w_img*0.25):int(w_img*0.75)]
        center_edges = cv2.Canny(center_strip, 100, 200)
        center_density = np.sum(center_edges > 0) / (center_strip.shape[0] * center_strip.shape[1])
        
        return {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2),
            "contrast": round(float(contrast), 2),
            "edge_density": round(float(edge_density), 4),
            "avg_saturation": round(float(avg_saturation), 2),
            "clip_product_labels": clip_labels,
            "yolo_fallback_labels": list(set(yolo_labels)),
            "text_regions_count": text_regions_detected,
            "has_central_focus": bool(center_density > 0.01),
            "is_studio_lighting": bool(brightness > 180 and edge_density < 0.04),
            "image_dimensions": f"{img.shape[1]}x{img.shape[0]}"
        }
