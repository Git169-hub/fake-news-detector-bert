# Fake News Detector — DistilBERT

A fine-tuned DistilBERT model that classifies news articles as FAKE or REAL.
Deployed as an interactive Gradio app on HuggingFace Spaces.

## Live Demo

👉 [Try it here](https://huggingface.co/spaces/RazakAIhub/fake-news-detector)

## Model

👉 [RazakAIhub/distilbert-fake-news-classifier](https://huggingface.co/RazakAIhub/distilbert-fake-news-classifier)

> 51+ downloads on HuggingFace Model Hub

## Performance

| Metric | Score |
|---|---|
| Test Accuracy | 99.15% |
| F1 Score | 99.15% |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Dataset Size | 24,353 articles |
| Training Hardware | Google Colab T4 GPU |

## Dataset

- **Source:** GonzaloA/fake_news (HuggingFace)
- **Size:** 24,353 articles
- **Labels:** FAKE (0), REAL (1)

## How It Works

1. Input text is tokenized using DistilBERT tokenizer (max 256 tokens)
2. DistilBERT encodes the text using self-attention across all tokens
3. The CLS token representation is passed to a classification head
4. Model outputs probability scores for FAKE and REAL

## How to Use

```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="RazakAIhub/distilbert-fake-news-classifier"
)

result = pipe("NASA confirms water ice found on the moon's surface.")
print(result)
# [{'label': 'REAL', 'score': 0.997}]
```

## How to Run Locally

```bash
git clone https://github.com/Git169-hub/fake-news-detector-bert
cd fake-news-detector-bert
pip install -r requirements.txt
python app.py
```

## Known Limitation

This model detects **writing style**, not factual accuracy.
It was trained on Reuters-style articles (REAL) vs partisan/opinion content (FAKE).
A well-written false article may still be classified as REAL.

## Tech Stack

| Component | Tool |
|---|---|
| Model | DistilBERT (distilbert-base-uncased) |
| Framework | PyTorch + HuggingFace Transformers |
| Deployment | Gradio on HuggingFace Spaces |
| Training | Google Colab T4 GPU |

## Author

Razak Shaik | VIT-AP University | CS Final Year
[HuggingFace](https://huggingface.co/RazakAIhub) | [LinkedIn](https://www.linkedin.com/in/shaik-razak-6493b7257) | [GitHub](https://github.com/Git169-hub)
