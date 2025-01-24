from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# We will use distilRoberta-financial-sentiment a fine-tuned version of distilroberta-base on the financial_phrasebank dataset
def distill_roberta_classify_sentiment(article):
    sentiment_result = sentiment_pipeline(article)[0]
    label = sentiment_result['label']
    
    if label == "positive":
        return 1
    elif label == "negative":
        return -1
    else:
        return 0