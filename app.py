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
    features = [[data.variance, data.skewness, data.curtosis, data.entropy]]

    res = model.predict(features)
    pred = "Fake Bank Note" if res >= 0.5 else "Bank Note OK"

    return {
        "Result":float(res[0]),
        "prediction":pred,
    }

@app.get("/")
def sample():
    return {
        "message":"model is working"
    }


if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)

