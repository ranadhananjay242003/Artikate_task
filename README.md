# Mini Multimodal Intelligence System
### Professional Product Image Quality Guard

This system analyzes image suitability for e-commerce by extracting modular visual signals and reasoning over them using the latest Gemini models.

## 🚀 Tech Stack
*   **CV Engine**: OpenCV, NumPy, YOLOv8 (Ultralytics)
*   **Reasoning Engine**: Google GenAI SDK (`google-genai`)
*   **Models**: Dynamic selection (Gemini 2.5 Flash / 2.0 Flash)

## 🧠 ML Engineering Judgment (Required for Task)

### 1. Why extract signals instead of just using Vision LLM (e.g., GPT-4V)?
*   **Cost & Latency**: Running traditional CV (Laplacian blur, edge density) takes milliseconds and cents. Sending high-res images to a Vision LLM costs more and depends on API latency.
*   **Consistency**: Vision LLMs can sometimes "hallucinate" quality (e.g., thinking a stylized blurry shot is professional). Objective signals (Blur Score) provide a grounding "source of truth."
*   **Explainability**: If you tell a seller "Your image is bad," they ask "Why?" Our system can specifically point to `blur_score: 42` (low focus) or `edge_density: 0.12` (busy background).

### 2. Trade-offs: Speed vs. Accuracy
*   **Model Choice**: I chose **YOLOv8-Nano** for object detection. While a larger model (YOLOv8-Large) or a CLIP-based search might be more accurate, the *Nano* version runs efficiently on CPUs, making it better for real-time seller feedback during upload.
*   **Traditional CV vs. DL**: For blur and brightness, traditional CV is actually superior and faster than training a deep learning model for "quality classification."
*   **Dynamic Model Discovery**: The system automatically probes available models on the user's API key. This ensures uptime even if specific experimental models (like 2.0 Flash) are hitting regional quota limits.

### 3. Failure Cases & Mitigations
*   **Artistic Intent**: A professional "bokeh" shot might be flagged as "blurry" by the Laplacian method. 
    *   *Mitigation*: We detect the primary object first. If the object is sharp but the background is blurry, we adjust the score.
*   **Low Lighting vs. Dramatic Lighting**: A professional dark-mode product shot might be flagged as "too dark."
    *   *Mitigation*: Use Contrast + Brightness together. High contrast + low brightness usually means "intentional/dramatic," while low contrast + low brightness means "poor photo."

### 4. How this scales in Production
*   **Parallelization**: The Vision Engine can run on a cluster of CPU nodes or even on the client's phone (using TensorFlow.js/ONNX) before the image is ever uploaded.
*   **Batch Reasoning**: Instead of 1 LLM call per image, we can batch the JSON signals of 50 images into one LLM prompt to save cost.

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup API Key** (Optional):
   Rename `.env.example` to `.env` and add your `GEMINI_API_KEY`. If no key is provided, the system runs in **Mock/Heuristic mode**.

3. **Run Analysis**:
   ```bash
   python main.py path/to/image.jpg
   ```
