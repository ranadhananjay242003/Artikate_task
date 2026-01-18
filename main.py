import argparse
import sys
import json
from app.vision_engine import VisionEngine
from app.reasoning_engine import ReasoningEngine

def main():
    parser = argparse.ArgumentParser(description="Multimodal Intelligence System - Product Image Analyzer")
    parser.add_argument("image_path", help="Path to the image to analyze")
    args = parser.parse_args()

    print(f"--- Analyzing Image: {args.image_path} ---")

    # 1. Vision Intelligence (Feature Extraction)
    print("[1/2] Extracting visual signals...")
    try:
        vision = VisionEngine()
        features = vision.analyze_image(args.image_path)
        print("Signals Extracted:")
        print(json.dumps(features, indent=4))
    except Exception as e:
        print(f"Error in Vision Engine: {e}")
        sys.exit(1)

    # 2. Reasoning Layer (LLM Judgment)
    print("\n[2/2] Reasoning over signals...")
    reasoning = ReasoningEngine()
    verdict = reasoning.reason(features)

    print("\n--- FINAL VERDICT ---")
    print(json.dumps(verdict, indent=4))

if __name__ == "__main__":
    main()
