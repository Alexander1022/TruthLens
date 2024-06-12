import sys
import os
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from get_result import Classifier

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    predicted_class: int
    predicted_proba: List[float]
    answer: str

classifier = Classifier()

@app.get("/")
def read_root() -> Union[dict, str]:
    return {"message": "Hello there. This is TruthLens API."}

@app.get("/health")
def health_check() -> Union[dict, str]:
    return {"status": "ok"}

@app.post("/answer", response_model=ClassificationResult)
def answer(sentence: TextRequest):
    predicted_class, predicted_proba, answer = classifier.classify_sentence_with_proba(sentence.text)
    print(f"Predicted class: {predicted_class}, Probability: {predicted_proba}")
    return ClassificationResult(predicted_class=predicted_class, predicted_proba=predicted_proba, answer=answer)
