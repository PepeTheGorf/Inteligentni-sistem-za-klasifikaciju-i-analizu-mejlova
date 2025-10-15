from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

# Load model & tokenizer once when the server starts
MODEL_PATH = "./fine_tuned_spam_classifier"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class TextInput(BaseModel):
    subject: str
    content: str

#uvicorn distil_bert_cross:app --reload
@app.post("/predict/")
async def predict(input_data: TextInput):
    text = input_data.subject + " " + input_data.content
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()

    label = "spam" if pred_class == 1 else "ham"
    confidence = round(probs[0][pred_class].item(), 4) * 100

    return {
        "label": label,
        "confidence": confidence
    }
