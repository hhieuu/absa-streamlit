import logging
from typing import List
from source.classes import Sentence

def init_local_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(filename)s : %(lineno)d - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def parse_multiple_sentences(text):
    splitted_text = text.split("\n")
    return [sent.strip() for sent in splitted_text]


def _construct_output_text(sentence: Sentence):
    sentence_text = f"**Sentence**: {sentence.text}\n"
    aspects = sentence.aspects
    aspect_texts = []
    for i, aspect in enumerate(aspects):
        _txt = f"{i + 1}. **Aspect**: {aspect.text} => {aspect.sentiment}: {aspect.scores}\n"
        aspect_texts.append(_txt)
        
    if aspect_texts:
        aspect_text = "".join(aspect_texts)
    else:
        aspect_text = "- No aspect extracted!\n"
        
    return sentence_text + aspect_text


def construct_output_text(sentences: List[Sentence]):
    all_sentences_texts = []
    for sentence in sentences:
        one_sentence_text = _construct_output_text(sentence)
        all_sentences_texts.append(one_sentence_text)
        
    if all_sentences_texts:
        return "".join(all_sentences_texts)
    else:
        return "No sentence to parse!"
    