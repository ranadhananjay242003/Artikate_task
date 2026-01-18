# Mini Multimodal Intelligence System
### Professional Product Image Quality Guard

This system analyzes image suitability for e-commerce by extracting modular visual signals and reasoning over them using the latest Gemini models.

## 🚀 Tech Stack
*   **CV Engine**: OpenCV, NumPy, YOLOv8 (Ultralytics)
*   **Product Classification**: CLIP (OpenAI Vision-Language Model)
*   **Reasoning Engine**: Google GenAI SDK (`google-genai`)
*   **Models**: Dynamic selection (Gemini 2.5 Flash / 2.0 Flash)
*   **Web Interface**: Gradio

## 🎯 System Architecture

### Stage 1: Vision Intelligence Layer
Extracts objective signals using traditional Computer Vision and modern Deep Learning:
*   **Blur Detection**: Variance of Laplacian (OpenCV)
*   **Exposure Analysis**: Brightness, Contrast, Overexposure detection
*   **Background Quality**: Edge density for clutter detection
*   **Studio Lighting Detection**: High-key photography recognition
*   **Product Classification**: CLIP zero-shot classification (20+ e-commerce categories)
*   **Text Detection**: OCR region identification

### Stage 2: LLM Reasoning Layer
Uses structured JSON signals to make high-level qualitative judgments:
*   Analyzes extracted signals in context
*   Corrects misclassifications from CV models
*   Provides explainable quality scores
*   Generates actionable feedback

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Key (Optional)
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```
If no key is provided, the system runs in **Mock/Heuristic mode**.

### 3. Run the Application

#### **Option A: Web Interface (Recommended)** 🌐
```bash
python app_ui.py
```
Then open **`http://localhost:7860`** in your browser.

**Features:**
- 📤 Drag-and-drop image upload
- 🔬 Expandable "Extracted Visual Signals" panel
- 🧠 "Final Verdict" with LLM reasoning
- 🎨 Professional gradient UI with syntax-highlighted JSON

#### **Option B: Command Line** 💻
```bash
python main.py path/to/image.jpg
```

---

## 📊 Output Format

```json
{
  "image_quality_score": 0.95,
  "issues_detected": [],
  "detected_objects": ["athletic footwear", "sneakers"],
  "text_detected": [],
  "llm_reasoning_summary": "High-quality studio shot with excellent sharpness...",
  "final_verdict": "Suitable for professional e-commerce use",
  "confidence": 0.95
}
```

---

## 📁 Project Structure

```
Artikate_task/
├── app/
│   ├── vision_engine.py      # Pre-LLM feature extraction
│   └── reasoning_engine.py   # LLM-based quality assessment
├── app_ui.py                  # Gradio web interface
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🔒 Security Note

The `.env` file containing your API key is **not committed** to version control. Always use `.gitignore` to protect sensitive credentials.

---

## 📝 License

This project was created as part of the Artikate Studio Machine Learning Engineer evaluation task.
