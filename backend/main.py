from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained("heyunmani/commentSense").to(device)
tokenizer = AutoTokenizer.from_pretrained("heyunmani/commentSense")
model.eval()

# FastAPI app
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def get_prediction(request: TextRequest):
    inputs = tokenizer(
        request.text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    label_map = {0: "complaint", 1: "demands", 2: "praise", 3: "questions"}
    return {"prediction": label_map.get(pred_class, "Unknown")}
