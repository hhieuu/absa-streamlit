import sys
import os
import requests
import json
from datetime import datetime

import streamlit as st

import config
import utils
from source.classes import Aspect, Sentence

logger = utils.init_local_logger('streamlit-app') # cloud logger

# simple helper function
def make_absa_api_request(text: str):
    try:
        params = {"text": text}
        response = requests.post(config.ABSA_API_ADDRESS, params=params)
        status_code = response.status_code
        response_data = json.loads(response.content)
            
    except:
        logger.exception("Error happens while requesting for prediction")
        response_data = {}
        
    return response_data

    
# Streamlit app
st.title(config.APP_TITLE)
markdown_text = """
----

## Hello there!
Welcome to a simple Streamlit app to demonstrate Aspect Term Extraction and Sentiment Analysis!

## Some details:

Aspect Terms is extracted by parsing the dependency tree and get the noun phrases. Currently, only nouns (phrases) are extracted. Dependency tree is generated using Spacy NLP Package

Sentiment for each aspect term is extracted using model from this GitHub repo: [Link here](https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis)
- This model is based on finetuning a pretrained "bert-base-uncased" model.
- Data for training are from SemEval-2014 task4. Train data from all categories are joined and trained together. Categories used are:
    - Laptops:
        - train: 2327
        - test: 636
     - Restaurants:
        - train: 3602
        - test: 1119
    - Twitter:
        - train: 6247
        - test: 691

"""
st.markdown(markdown_text)
# demo zone
st.header("Demo zone!")
st.subheader("Input text for parsing")
input_text = st.text_area(
    label="Input text here",
    value="Great starter career for those without a family. Requires a lot of traveling if you're a consultant. The pay and benefits are good."
)

# get and parse input text
st.subheader("Parse results:")
response_data = make_absa_api_request(input_text)
output_markdown = ""
for i, aspect in enumerate(response_data.get("aspects", [])):
    output_markdown += f"{i + 1}. **Aspect**: {aspect['aspect']} => **{aspect['sentiment']}**\n"
    
st.markdown(output_markdown)
