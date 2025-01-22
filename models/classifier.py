from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# We will use distilRoberta-financial-sentiment a fine-tuned version of distilroberta-base on the financial_phrasebank dataset
def distill_roberta_classify_sentiment(article):

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    sentiment_result = sentiment_pipeline(article)[0]
    label = sentiment_result['label']
    
    if label == "positive":
        return 1
    elif label == "negative":
        return -1
    else:
        return 0