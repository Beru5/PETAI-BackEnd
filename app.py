from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Tentukan folder download NLTK ke folder /tmp milik Vercel
nltk_data_dir = "/tmp/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download secara aman di lingkungan serverless
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

app = FastAPI(
    title="API PETAI",
    description="PEndeteksi Teks AI",
    version="1.0.0"
)

try:
    model = joblib.load("model_dir/model_petai.joblib")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model. Pastikan file model tersedia. Error: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/api/predict")
async def predict_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")

    cleaned_text = clean_text(req.text)
    
    pred_label = model.predict([cleaned_text])[0]
    pred_proba = model.predict_proba([cleaned_text])[0]

    if pred_label == 1.0:
        result_class = "AI"
        confidence = float(pred_proba[1] * 100) 
    else:
        result_class = "Human"
        confidence = float(pred_proba[0] * 100)

    return {
        "status": "success",
        "result": result_class,
        "confidence_percentage": round(confidence, 2),
        "details": {
            "human_probability": round(pred_proba[0] * 100, 2),
            "ai_probability": round(pred_proba[1] * 100, 2)
        }
    }