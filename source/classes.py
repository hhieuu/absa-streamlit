from typing import List, Any

from transformers import BertTokenizer
from spacy.language import Language

import config


tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)


# data classes
class Aspect:
    def __init__(
        self,
        text: str,
        scores: List[float] = None,
        label: int = None,
        sentiment: str = None,
    ):
        self.text = text
        self.tokens = tokenizer.tokenize(self.text)
        self.scores = scores
        self.label = label
        self.sentiment = sentiment
        
    def __dict__(self):
        return {
            "aspect": self.text,
            "scores": self.scores,
            "label": self.label,
            "sentiment": self.sentiment 
        }
        
    def to_dict(self):
        return self.__dict__()
        
    def __str__(self):
        if isinstance(self.scores, list):
            score = max(self.scores)
        else:
            score = None
            
        return_string = (
            f"Aspect("
            f"text={self.text}, "
            f"sentiment={self.sentiment}, "
            f"label={self.label}, "
            f"score={score})"
        )
        
        return return_string
    
    def __repr__(self):
        return self.__str__
    
    
class Sentence:
    def __init__(
        self,
        text: str,
        doc: Language = None,
        aspects: List[Aspect] = None,
    ):
        self.text = text
        self.tokens = tokenizer.tokenize(self.text)
        self.doc = doc
        if not aspects:
            self.aspects = []
        else:
            self.aspects = aspects
            
    def __dict__(self):
        return {
            "text": self.text,
            "aspects": [asp.to_dict() for asp in self.aspects]
        }
        
    def to_dict(self):
        return self.__dict__()
            
    def __str__(self):
        aspect_string = "(" + "\n\t\t".join([asp.__str__() for asp in self.aspects]) + ")"
        return_string = (
            "Sentence(\n"
            f"\ttext={self.text},\n"
            f"\taspects={aspect_string}"
            f"\t)"
        )
        return return_string
    
    def __repr__(self):
        return self.__str__