import gradio as gr
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Load ML model artifacts
model = joblib.load("crop_rf_model.pkl")
scaler = joblib.load("crop_scaler.pkl")
le = joblib.load("crop_label_encoder.pkl")

# Gemini model configs
generation_config = {
    "temperature": 0.5,
    "max_output_tokens": 1024,
}
text_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
image_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-image-preview",
)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for input validation
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def create_placeholder_image(crop_name):
    """Create a simple placeholder image when image generation fails"""
    img = Image.new('RGB', (300, 200), color='lightgray')
    return img

# FastAPI endpoint for crop prediction
@app.post("/predict")
def predict_crop(data: CropInput):
    sample = [data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]
    scaled = scaler.transform([sample])
    probs = model.predict_proba(scaled)[0]
    
    top_idx = np.argsort(probs)[-3:][::-1]
    crops = le.inverse_transform(top_idx)
    
    recommendations = []
    for j in range(len(top_idx)):
        crop_name = crops[j]
        probability = float(probs[top_idx[j]])
        
        # Generate description using text model
        text_prompt = f"""
        Provide a brief description of the general health benefits of {crop_name} for humans and farming tips.
        """
        try:
            text_response = text_model.generate_content([text_prompt])
            description = text_response.text
        except Exception as e:
            description = f"Health benefits and farming information for {crop_name} could not be generated."
        
        # Generate image using image model
        image_prompt = f"Generate a high-quality image of {crop_name} for farmers."
        image_data = None
        mime_type = "image/png"
        
        try:
            image_response = image_model.generate_content([image_prompt])
            for part in image_response.candidates[0].content.parts:
                if hasattr(part, 'inline_data'):
                    image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                    mime_type = part.inline_data.mime_type
                    break
        except Exception as e:
            print(f"Image generation failed for {crop_name}: {e}")
            image_data = None
        
        recommendations.append({
            "crop": crop_name,
            "probability": probability,
            "description": description,
            "image_data": image_data,
            "mime_type": mime_type
        })
        
    return {"recommendations": recommendations}

# Gradio interface function - FIXED VERSION
def get_recommendations(N, P, K, temperature, humidity, ph, rainfall):
    payload = {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }
    
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()
        outputs = []
        
        for rec in res["recommendations"]:
            if rec["image_data"]:
                # Convert base64 to PIL Image
                img_bytes = base64.b64decode(rec["image_data"])
                img = Image.open(io.BytesIO(img_bytes))
            else:
                # Create placeholder image instead of None
                img = create_placeholder_image(rec["crop"])
            
            # Format the description text
            caption = f"**{rec['crop']}** (Confidence: {rec['probability']:.2%})\n\n{rec['description']}"
            
            outputs.append((img, caption))
        
        return outputs
        
    except Exception as e:
        # Return error message if API call fails
        error_img = create_placeholder_image("Error")
        return [(error_img, f"Error occurred: {str(e)}")]

# Alternative version using only valid images
def get_recommendations_filter_none(N, P, K, temperature, humidity, ph, rainfall):
    payload = {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }
    
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()
        outputs = []
        
        for rec in res["recommendations"]:
            # Only add items that have valid image data
            if rec["image_data"]:
                img_bytes = base64.b64decode(rec["image_data"])
                img = Image.open(io.BytesIO(img_bytes))
                caption = f"**{rec['crop']}** (Confidence: {rec['probability']:.2%})\n\n{rec['description']}"
                outputs.append((img, caption))
            else:
                # Skip items without images, or add text-only description
                print(f"Skipping {rec['crop']} - no image generated")
        
        if not outputs:
            # If no images were generated, return placeholder
            placeholder_img = create_placeholder_image("No images available")
            outputs.append((placeholder_img, "No crop images could be generated at this time."))
        
        return outputs
        
    except Exception as e:
        error_img = create_placeholder_image("Error")
        return [(error_img, f"Error occurred: {str(e)}")]

# Launch Gradio interface
if __name__ == "__main__":
    interface = gr.Interface(
        fn=get_recommendations,  # Use get_recommendations_filter_none as alternative
        inputs=[
            gr.Number(label="Nitrogen (N)"), 
            gr.Number(label="Phosphorus (P)"),
            gr.Number(label="Potassium (K)"),
            gr.Number(label="Temperature (°C)"), 
            gr.Number(label="Humidity (%)"),
            gr.Number(label="pH"), 
            gr.Number(label="Rainfall (mm)")
        ],
        outputs=gr.Gallery(
            label="Top 3 Crop Recommendations",
            columns=1,
            object_fit="contain",
            height="auto"
        ),
        title="🌱 Smart Crop Recommendation System",
        description="Enter soil and environmental parameters to get personalized crop recommendations with health benefits and farming tips.",
        
    )
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )