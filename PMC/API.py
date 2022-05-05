from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from joblib import load

app = FastAPI()
class DataModelapp(BaseModel):
    temperature: float
    humidity: float
    ph: float
class ListPrediccion(BaseModel):
    datos:List[DataModelapp]=None
@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/Data/predict")
async def make_prediction(dataModel: DataModelapp):
    
    df = pd.DataFrame(dataModel.dict(),index=[0])
    model = load("pipeline.joblib")
    result = model.predict(df)
    resultado=result[0]
    return {"Tiempo de expectativa de vida: ": resultado}

@app.post("/Data/predict/list")
async def make_predictionsList(dataList: ListPrediccion):
    lista_datos=dataList.dict()['datos']
    model = load("pipeline.joblib")
    respuestastr=''
    for i in lista_datos:
        df = pd.DataFrame(i,index=[0])
        result = model.predict(df)
        resultado=result[0]
        respuestastr=respuestastr+' , '+str(resultado)
    return {"Tiempos de expectativa de vida de cada set de datos: ": respuestastr}