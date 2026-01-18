from google import genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

class ReasoningEngine:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.client = None
            self.mock_mode = True
        else:
            self.client = genai.Client(api_key=api_key)
            self.mock_mode = False
            self.model_name = self._get_best_working_model()

    def _get_best_working_model(self):
        try:
            available_models = [m.name for m in self.client.models.list()]
            priorities = ['models/gemini-2.5-flash', 'models/gemini-1.5-flash', 'models/gemini-2.0-flash-exp']
            for m in priorities:
                if m in available_models: return m
            return 'gemini-1.5-flash'
        except: return 'gemini-1.5-flash'

    def reason(self, features):
        if self.mock_mode:
            return self._mock_reasoning(features)

        prompt = f"""
        You are analyzing visual signals from a product image for e-commerce quality.

        EXTRACTED SIGNALS:
        - Resolution: {features['image_dimensions']}
        - Blur Score: {features['blur_score']} (>100 is sharp)
        - Brightness: {features['brightness']} (150-220 ideal for studio)
        - Contrast: {features['contrast']}
        - Background Cleanliness: {features['edge_density']} (<0.04 is professional)
        - Studio Lighting Detected: {features['is_studio_lighting']}
        - CLIP Product Classification: {features['clip_product_labels']} (High-accuracy e-commerce labels)
        - YOLO Fallback Labels: {features['yolo_fallback_labels']} (General object detection)
        - Text Regions Found: {features['text_regions_count']}
        - Central Product Focus: {features['has_central_focus']}

        CRITICAL INSTRUCTION:
        Use the "CLIP Product Classification" as your PRIMARY source for product identification.
        Only reference YOLO labels if CLIP returned empty results.

        OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
        {{
            "image_quality_score": 0.95,
            "issues_detected": ["list any quality issues, or empty array if none"],
            "detected_objects": ["corrected product names, NOT the raw CV labels"],
            "text_detected": ["any text/watermarks found, or empty array"],
            "llm_reasoning_summary": "Brief explanation of your verdict",
            "final_verdict": "Suitable for professional e-commerce use" or "Not suitable...",
            "confidence": 0.95
        }}
        """

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            text = response.text
            # Strip markdown code blocks if present
            text = text.replace('```json', '').replace('```', '')
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            return json.loads(text[start_idx:end_idx])
        except Exception as e:
            return {"error": str(e), "fallback": self._mock_reasoning(features)}

    def _mock_reasoning(self, features):
        detected = features.get('clip_product_labels', []) or features.get('yolo_fallback_labels', [])
        return {
            "image_quality_score": 0.5,
            "issues_detected": ["Mock Mode Active"],
            "detected_objects": detected,
            "text_detected": [],
            "llm_reasoning_summary": "Heuristic fallback.",
            "final_verdict": "Review required.",
            "confidence": 0.5
        }
