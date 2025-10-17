from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification
)
import torch

app = FastAPI()

DISTIL_MODEL_PATH = "fine_tuned_spam_classifier"
ROBERTA_MODEL_PATH = "roberta_cross/content/fine_tuned_roberta_spam_classifier"
BERT_MODEL_PATH = "bert_cross/content/fine_tuned_bert_spam_classifier"

models = {
    "distilbert": {
        "tokenizer": DistilBertTokenizer.from_pretrained(DISTIL_MODEL_PATH),
        "model": DistilBertForSequenceClassification.from_pretrained(DISTIL_MODEL_PATH).eval(),
    },
    "roberta": {
        "tokenizer": RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH),
        "model": RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH).eval(),
    },
    "bert": {
        "tokenizer": BertTokenizer.from_pretrained(BERT_MODEL_PATH),
        "model": BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH).eval(),
    }
}

def render_form(result_html: str = ""):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Classifier</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 40px auto;
                background-color: #f9f9f9;
            }}
            h2 {{
                text-align: center;
                color: #333;
            }}
            form {{
                display: flex;
                flex-direction: column;
                gap: 12px;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            textarea, input[type=text], select {{
                width: 100%;
                padding: 10px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }}
            button {{
                padding: 10px;
                font-size: 16px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            .result {{
                margin-top: 20px;
                font-size: 18px;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                flex-direction: column;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
            }}
            .spam {{
                background-color: #ffcccc;
                color: #a60000;
                border: 1px solid #ff9999;
            }}
            .ham {{
                background-color: #ccffcc;
                color: #006600;
                border: 1px solid #99ff99;
            }}
        </style>
    </head>
    <body>
        <h2>Spam Classifier</h2>
        <form action="/predict_form/" method="post">
            <label>Model:</label>
            <select name="model_name" required>
                <option value="roberta">RoBERTa</option>
                <option value="bert">BERT-base</option>
                <option value="distilbert">DistilBERT</option>
            </select>

            <label>Subject:</label>
            <input type="text" name="subject" required>

            <label>Content:</label>
            <textarea name="content" rows="6" required></textarea>

            <button type="submit">Check</button>
        </form>
        {result_html}
    </body>
    </html>
    """


# uvicorn BERT_classifier_API:app --reload

@app.get("/", response_class=HTMLResponse)
async def form_page():
    return render_form()


@app.post("/predict_form/", response_class=HTMLResponse)
async def predict_form(
    model_name: str = Form(...),
    subject: str = Form(...),
    content: str = Form(...),
):
    text = subject + " " + content

    model_info = models.get(model_name)
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()

    label = "spam" if pred_class == 1 else "ham"
    confidence = round(probs[0][pred_class].item(), 4) * 100
    color_class = "spam" if label == "spam" else "ham"

    result_html = f"""
    <div class='result {color_class}'>
        <b>Model:</b> {model_name.upper()} <br>
        <b>Prediction:</b> {label.upper()} <br>
        <b>Confidence:</b> {confidence:.2f}%
    </div>
    """

    return render_form(result_html)
