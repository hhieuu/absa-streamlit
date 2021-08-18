import os

# keyword extraction
INCLUDE_STOPWORDS = ["lot", "lots", "alot", "thing", "user"]
EXCLUDE_STOPWORDS = ["no", 'not', "n't", "n\'t", "nt"]
SPACY_DISABLE_COMPONENTS = ['ner']

# ABSA
PRETRAINED_MODEL_NAME = "bert-base-uncased"
CHECKPOINT_PATH = "./checkpoints"
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# app
APP_TITLE = "Aspect Terms Extraction and Sentiment Analysis Demo App"
ABSA_API_ADDRESS = os.environ.get("ABSA_API_ADDRESS")
