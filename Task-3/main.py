from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Initialize FastAPI app
app = FastAPI()

# HTML templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  age: int = Form(...),
                  sex: int = Form(...),
                  cp: int = Form(...),
                  trestbps: int = Form(...),
                  chol: int = Form(...),
                  fbs: int = Form(...),
                  restecg: int = Form(...),
                  thalach: int = Form(...),
                  exang: int = Form(...),
                  oldpeak: float = Form(...),
                  slope: int = Form(...),
                  ca: int = Form(...),
                  thal: int = Form(...)):
    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }
    df = pd.DataFrame([data])
    df = df[model_columns]
    prediction = model.predict(df)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
