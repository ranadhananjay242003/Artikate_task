from google import genai
import os
import json
from dotenv import load_dotenv

# Ensure we load the .env file from the project root
load_dotenv()

class ReasoningEngine:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment or .env file.")
            print("Reasoning engine will operate in 'mock' mode.")
            self.client = None
            self.mock_mode = True
        else:
            self.client = genai.Client(api_key=api_key)
            self.mock_mode = False
            self.model_name = self._get_best_working_model()

    def _get_best_working_model(self):
        """Finds the best available model that actually has quota for your key."""
        try:
            available_models = [m.name for m in self.client.models.list()]
            
            # Priority list for models based on what we saw in your list
            priorities = [
                'models/gemini-2.5-flash',  # Top priority since it appeared for you
                'models/gemini-1.5-flash',
                'models/gemini-2.0-flash-exp',
                'models/gemini-1.5-pro',
                'models/gemini-2.5-pro'
            ]
            
            # Filter prioritized models that are actually in the available list
            to_test = [p for p in priorities if p in available_models]
            
            # If none of our priorities are found, just test everything that looks like an LLM
            if not to_test:
                to_test = [m for m in available_models if 'gemini' in m.lower()]

            print("DEBUG: Testing model quota...")
            for model_name in to_test:
                try:
                    # Quick tiny test to see if quota > 0
                    self.client.models.generate_content(
                        model=model_name,
                        contents="hi",
                        config={'max_output_tokens': 1}
                    )
                    print(f"DEBUG: Found working model: {model_name}")
                    return model_name
                except Exception as e:
                    # If it's a 429/Resource Exhausted, skip it
                    if "429" in str(e) or "quota" in str(e).lower():
                        continue
                    # If it's something else, keep it in mind
                    continue

            return 'gemini-1.5-flash' # Final fallback
        except Exception as e:
            print(f"DEBUG: Error listing models: {e}")
            return 'gemini-1.5-flash'

    def reason(self, features):
        """Uses LLM to reason over visual signals and provide a structured verdict."""
        if self.mock_mode:
            return self._mock_reasoning(features)

        prompt = f"""
        You are an AI Product Quality Specialist for a premium E-commerce platform.
        Analyze the following visual signals extracted from a product image:

        SIGNALS:
        - Image Resolution: {features['image_dimensions']}
        - Blur Score: {features['blur_score']} (Threshold: < 100 is blurry)
        - Brightness: {features['brightness']} (Scale 0-255, 100-200 is ideal. >230 is overexposed)
        - Contrast: {features['contrast']} (Higher is generally better for products)
        - Color Saturation: {features['avg_saturation']} (Dull < 30, Vibrant > 70)
        - Edge Density: {features['edge_density']} (Proxy for background clutter: > 0.05 is busy)
        - Detected Objects: {', '.join(features['detected_objects']) if features['detected_objects'] else 'None detected'}

        TASK:
        1. Evaluate if this image is suitable for a professional product listing.
        2. Identify specific issues based ONLY on the signals provided.
        3. Provide a structured JSON response.

        JSON FORMAT:
        {{
            "is_professional": boolean,
            "quality_score": float (0.0 to 1.0),
            "issues": [string],
            "reasoning_summary": "Short explanation of why we reached this verdict based on the signals",
            "primary_signal_influence": "Which signal (e.g., Blur, Overexposure) mattered most here?"
        }}

        Return ONLY the JSON.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            text = response.text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            return json.loads(text[start_idx:end_idx])
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"error": str(e), "fallback": self._mock_reasoning(features)}

    def _mock_reasoning(self, features):
        """Fallback mock reasoning for demo purposes without API key."""
        issues = []
        if features['blur_score'] < 100: 
            issues.append("Image appears out of focus/blurry")
        if features['brightness'] < 50: 
            issues.append("Lighting is too dark")
        if features['brightness'] > 230:
            issues.append("Image is overexposed/too bright")
        if not features['detected_objects']:
            issues.append("No clear product detected in frame")
        if features['edge_density'] > 0.08: 
            issues.append("Background is too cluttered/busy")
        
        is_prof = len(issues) == 0
        return {
            "is_professional": is_prof,
            "quality_score": round(1.0 - (len(issues) * 0.25), 2),
            "issues": issues,
            "reasoning_summary": f"Analyzed signals: Brightness={features['brightness']}, Objects={len(features['detected_objects'])}. Issues found: {len(issues)}",
            "primary_signal_influence": "Exposure and Object Detection" if not features['detected_objects'] else "Lighting quality"
        }
