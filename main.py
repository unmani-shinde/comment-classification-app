from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
class MultiClassClassifier(nn.Module):
    def __init__(self):
        super(MultiClassClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 4)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)  
        return logits

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Download model weights from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="heyunmani/comment-classifier-app",  # 👈 replace with actual values
    filename="model_state.pth"
)

# Load model and weights
model = MultiClassClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()

# FastAPI app
app = FastAPI()

# Input schema
class TextRequest(BaseModel):
    text: str

# Prediction function
def predict(text: str):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    label_map = {0: "complaint", 1: "demands", 2: "praise", 3: "questions"}
    return label_map.get(pred_class, "Unknown")

@app.post("/predict")
def get_prediction(request: TextRequest):
    prediction = predict(request.text)
    return {"prediction": prediction}

@app.get("/")
def root():
    return {"message": "Welcome to the Comment Classifier API. Visit /docs to use it."}
