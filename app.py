from fastapi import FastAPI
import pickle,uvicorn
from pydantic import BaseModel

class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float


app = FastAPI()
pkl_in = open('bankAuthClassifier.pkl','rb')
model = pickle.load(pkl_in)

@app.post('/predict')
def predictor(data:BankNote):
    data = data.dict()
    var = data['variance']
    skew = data['skewness']
    curt = data['curtosis']
    ent = data['entropy']

    res = model.predict([[var,skew,curt,ent]])
    if res >= 0.5:
        pred = "Fake Bank Note"
    else:
        pred = "Bank Note OK"

    return {
        "prediction":pred,
    }

@app.get("/")
def sample():
    return {
        "message":"model is working"
    }


if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)

