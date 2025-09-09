import os
from pathlib import Path
import mimetypes
from dotenv import load_dotenv
import gradio as gr
import google.generativeai as genai

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in your environment/.env file.")
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Image+text capable model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# -----------------------------
# Helpers
# -----------------------------
def guess_mime_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    mt, _ = mimetypes.guess_type(file_path)
    return mt or "application/octet-stream"

def read_image_data(file_path: str):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    mime = guess_mime_type(file_path)
    if not mime.startswith("image/"):
        raise ValueError("Please upload a valid image file (jpg, jpeg, png, webp, bmp).")
    return {"mime_type": mime, "data": p.read_bytes()}

def build_prompt(language: str) -> str:
    # Plain, farmer-friendly; no markdown symbols, no bold; simple sections.
    return f"""
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

Please follow these strict rules when giving your response:

1.  first find out which crop or plant it is and highlight as heading Prefer common names for the plant and disease. Mention scientific names only if they are useful for clarity and highlight it like heading.  
2. Translate the entire response into the user’s chosen language: {language} .
3. Do not use * # or ** like gpt generated use best formating for headings and bold plant name and dieseases and dont allow spaces one line after other and provide properly use * if neccesary. 
4. Structure your response into clear sections:  
   - **Identification**: identify the plant and general name and diesease Describe the symptoms and signs observed.  
   - **Diagnosis**: Identify the disease or issue affecting the plant.  
   - **Causes**: Explain the potential causes of the disease, including environmental factors, pests, or pathogens.  
   - **Treatment Options**: Provide a list of effective treatment methods, including chemical, biological, and cultural practices.  
   - **Prevention Strategies**: Suggest measures to prevent future occurrences of the disease.  
   - **Additional Notes**: Include any other relevant information that could assist in managing plant health. 
5. Keep the tone friendly, supportive, and practical.  
6. End with a short single-line summary advice and high light points .  

Important note: Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Always consult with qualified agricultural experts before implementing any strategies or treatments.

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, strictly following this structured format.
""".strip()

def generate_gemini_response(image_path: str, language: str) -> str:
    image_data = read_image_data(image_path)
    prompt = build_prompt(language)
    resp = model.generate_content([prompt, image_data])
    text = getattr(resp, "text", None)
    if not text:
        return "Sorry, I could not generate a response for this image."
    return text.replace("**", "").replace("#", "")

# -----------------------------
# Inference Functions
# -----------------------------
def process_uploaded_file(file_path: str, language: str):
    try:
        if not file_path:
            return None, "No file uploaded.", None
        if not os.path.exists(file_path):
            return None, f"Error: File not found at {file_path}", None
        result = generate_gemini_response(file_path, language)
        return file_path, result, file_path
    except Exception as e:
        return None, f"Error: {e}", None

def reanalyze_with_saved_path(saved_path: str, language: str):
    try:
        if not saved_path or not os.path.exists(saved_path):
            return None, "Please upload an image first."
        result = generate_gemini_response(saved_path, language)
        return saved_path, result
    except Exception as e:
        return None, f"Error: {e}"

# -----------------------------
# UI (Gradio)
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("🌱 Plant Disease Detector (Multilingual, Farmer Friendly)")

    language_choice = gr.Dropdown(
        choices=[
            "English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi",
            "Bengali", "Gujarati", "Punjabi", "Malayalam", "Odia", "Assamese", "Urdu"
        ],
        value="English",
        label="Choose your language"
    )

    image_output = gr.Image(label="Uploaded Image", interactive=False)
    text_output = gr.Textbox(label="Analysis Result", lines=20)

    last_image_path = gr.State(value="")

    upload_input = gr.File(
        label="Upload Plant Image",
        file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
        type="filepath"
    )

    upload_input.change(
        fn=process_uploaded_file,
        inputs=[upload_input, language_choice],
        outputs=[image_output, text_output, last_image_path],
        queue=True,
        show_progress=True,
    )

    language_choice.change(
        fn=reanalyze_with_saved_path,
        inputs=[last_image_path, language_choice],
        outputs=[image_output, text_output],
        queue=True,
        show_progress=False,
    )

    # -----------------------------
    # Extra: Chatbot Functionality
    # -----------------------------
    gr.Markdown("💬 Ask doubts, clarifications, or suggestions (about uploaded crops or general plant health)")

    chatbot = gr.Chatbot(label="Crop/Plant Chat")
    msg = gr.Textbox(label="Type your question here")
    send_btn = gr.Button("Ask")

    def chat_with_model(message, history, saved_path, language):
        try:
            clarifying_prompt = f"""
You are an agricultural assistant.
Your job is to help farmers with all farming-related questions:
- Crop selection, soil types, climate conditions, and growing practices.
- Plant diseases, pests, and their management.
- Fertilizers, soil health, irrigation, and best agricultural practices.
- General guidance on farming techniques and productivity improvement.

Ignore only questions completely unrelated to farming.
Respond in {language}.

Farmer's question: {message}
"""
            if saved_path and os.path.exists(saved_path):
                image_data = read_image_data(saved_path)
                resp = model.generate_content([clarifying_prompt, image_data])
            else:
                resp = model.generate_content([clarifying_prompt])
            text = getattr(resp, "text", None) or "Sorry, I could not answer."
            history.append(("User", message))
            history.append(("Bot", text.replace("**", "").replace("#", "")))
            return history
        except Exception as e:
            history.append(("User", message))
            history.append(("Bot", f"Error: {e}"))
            return history

    send_btn.click(
        fn=chat_with_model,
        inputs=[msg, chatbot, last_image_path, language_choice],
        outputs=chatbot,
    )
    msg.submit(
        fn=chat_with_model,
        inputs=[msg, chatbot, last_image_path, language_choice],
        outputs=chatbot,
    )

# -----------------------------
# Launch
# -----------------------------
demo.launch(debug=True, server_port=7861)
