import joblib
import pandas as pd

from fastapi import FastAPI, UploadFile, File

from train import model

app = FastAPI(docs_url='/', title = 'Deploy DM BI MAster Puc Rio')

model = joblib.load('model/spine_model.pkl')