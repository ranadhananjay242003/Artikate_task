import gradio as gr
import json
from app.vision_engine import VisionEngine
from app.reasoning_engine import ReasoningEngine

# Initialize engines
vision = VisionEngine()
reasoning = ReasoningEngine()

def analyze_product_image(image):
    """Main analysis pipeline for the web interface."""
    if image is None:
        return "Please upload an image first.", ""
    
    try:
        # Save uploaded image temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        # Stage 1: Vision Intelligence
        print("Extracting visual signals...")
        features = vision.analyze_image(temp_path)
        
        # Stage 2: LLM Reasoning
        print("Reasoning over signals...")
        verdict = reasoning.reason(features)
        
        # Format outputs
        signals_json = json.dumps(features, indent=2)
        verdict_json = json.dumps(verdict, indent=2)
        
        return signals_json, verdict_json
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg

# Custom CSS for premium look
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 2em;
}
"""

# Build the interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 id='title'>🔍 Product Image Quality Guard</h1>")
    gr.Markdown("<p id='subtitle'>Professional E-commerce Image Analysis System</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Product Image")
            image_input = gr.Image(
                type="pil",
                label="Product Photo",
                height=400
            )
            analyze_btn = gr.Button("🚀 Analyze Image Quality", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📋 How it works:
            1. **Vision Engine** extracts objective signals (blur, lighting, objects)
            2. **CLIP Model** identifies product category
            3. **LLM Reasoning** provides final quality verdict
            """)
    
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            
            with gr.Accordion("🔬 Extracted Visual Signals", open=False):
                signals_output = gr.Code(
                    label="Pre-LLM Intelligence Layer",
                    language="json",
                    lines=15
                )
            
            with gr.Accordion("🧠 Final Verdict (LLM Reasoning)", open=True):
                verdict_output = gr.Code(
                    label="E-commerce Quality Assessment",
                    language="json",
                    lines=15
                )
    
    # Connect the button
    analyze_btn.click(
        fn=analyze_product_image,
        inputs=[image_input],
        outputs=[signals_output, verdict_output]
    )

if __name__ == "__main__":
    print("🚀 Starting Product Image Quality Guard...")
    print("📱 Open the URL below in your browser:")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )
