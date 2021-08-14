# stdlibs
import json

# 3rd party
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from typing import Optional

# own
import config
import utils
import helpers


# init logger
logger = utils.init_local_logger('prediction-api') # cloud logger

# init app
app = FastAPI(title='ABSA-API')


@app.get('/')
def hello():
    return "Welcome to ABSA API!"


# endpoint for job-classifier
@app.post('/predict')
async def absa(text: str):
    """
    Main entry point for ATE and ABSA
    """
    sentence = helpers.do_absa(text=text)
    result = sentence.to_dict()

    return result
