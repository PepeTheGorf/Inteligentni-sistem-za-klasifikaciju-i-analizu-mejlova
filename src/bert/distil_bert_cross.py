from fastapi import FastAPI, Form
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

#load model & tokenizer once when the server starts
MODEL_PATH = "./fine_tuned_spam_classifier"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class TextInput(BaseModel):
    subject: str
    content: str

@app.get("/", response_class=HTMLResponse)
async def form_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; }
            h2 { text-align: center; }
            form { display: flex; flex-direction: column; gap: 12px; }
            textarea, input[type=text] { width: 100%; padding: 8px; font-size: 14px; }
            button { padding: 10px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; font-size: 18px; }
        </style>
    </head>
    <body>
        <h2>Spam Classifier</h2>
        <form action="/predict_form/" method="post">
            <label>Subject:</label>
            <input type="text" name="subject" required>

            <label>Content:</label>
            <textarea name="content" rows="6" required></textarea>

            <button type="submit">Check</button>
        </form>
        {result}
    </body>
    </html>
    """
    return html_content.format(result="")


@app.post("/predict_form/", response_class=HTMLResponse)
async def predict_form(subject: str = Form(...), content: str = Form(...)):
    text = subject + " " + content
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()

    label = "spam" if pred_class == 1 else "ham"
    confidence = round(probs[0][pred_class].item(), 4) * 100

    result_html = f"""
    <div class='result'>
        <b>Prediction:</b> {label.upper()} <br>
        <b>Confidence:</b> {confidence:.2f}%
    </div>
    """

    #re-render the form with result below
    return await form_page().replace("{result}", result_html)