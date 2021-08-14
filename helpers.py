import logging
from spacy.language import Language
from typing import List

import torch
from transformers import BertTokenizer

import config
from model import extractor
import utils
from model.bert import (
    bert_ABSA,
    load_model
)
from source.classes import Aspect, Sentence


# initialize
logger = utils.init_local_logger(logger_name='helper')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# spacy
spacy_pipeline = extractor.initialize_spacy_pipeline(logger=logger)
# ABSA model
logger.info('Initializting ABSA model...')
tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
absa_model = bert_ABSA(config.PRETRAINED_MODEL_NAME).to(DEVICE)
absa_model = load_model(absa_model, 'checkpoints/bert_ABSA2.pkl', device=DEVICE)


# functions
def parse_text_for_aspects(text: str):
    doc = spacy_pipeline(text)
    aspects = extractor.parse_noun_chunks(
        doc, 
        do_lemma=False, 
        do_lower=False,
        get_verb=False,
        return_base_chunk=False
    )
    sentence = Sentence(
        text=text,
        doc=doc,
        aspects=[Aspect(asp) for asp in aspects]
    )
    return sentence


def do_absa_single_aspect(sentence_tokens: List[str], aspect_tokens: List[str]):
    # initiate wordpieces
    word_pieces = ['[cls]']
    word_pieces += sentence_tokens
    word_pieces += ['[sep]']
    word_pieces += aspect_tokens
    # construct ids tensor and segment tensor
    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    segment_tensor = [0] + [0] * len(sentence_tokens) + [0] + [1] * len(aspect_tokens)
    # push to appropriate device
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)
    # do ABSA
    with torch.no_grad():
        outputs = absa_model(input_tensor, None, None, segments_tensors=segment_tensor)
        _, pred_label = torch.max(outputs, dim=1)
        sentiment = config.SENTIMENT_MAPPING.get(pred_label[0].tolist(), "UNK")
    
    return outputs.tolist()[0], pred_label[0].tolist(), sentiment


def do_absa(text: str):
    sentence = parse_text_for_aspects(text)
    for aspect in sentence.aspects:
        outputs, pred_label, sentiment = do_absa_single_aspect(
            sentence.tokens,
            aspect.tokens
        )
        aspect.scores = outputs
        aspect.label = pred_label
        aspect.sentiment = sentiment
        
    return sentence
