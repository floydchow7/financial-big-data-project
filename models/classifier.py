from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
# We will use distilRoberta-financial-sentiment a fine-tuned version of distilroberta-base on the financial_phrasebank dataset
def load_sentiment_model(model_name = "mrm8448/distilroberta-finetuned-financial-news-sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #model = model.half()
    return model, tokenizer