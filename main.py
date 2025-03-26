# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

from .prediction import (
    predict_murmur,
    predict_murmur_timing,
    predict_murmur_quality,
    predict_murmur_grading,
    predict_murmur_shape,
    predict_murmur_pitch
)

app = FastAPI()

def handle_prediction(predict_function, file: UploadFile):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_function(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/murmur/")
async def predict_murmur_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur, file)

@app.post("/predict/timing/")
async def predict_timing_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur_timing, file)

@app.post("/predict/quality/")
async def predict_quality_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur_quality, file)

@app.post("/predict/grading/")
async def predict_grading_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur_grading, file)

@app.post("/predict/shape/")
async def predict_shape_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur_shape, file)

@app.post("/predict/pitch/")
async def predict_pitch_endpoint(file: UploadFile = File(...)):
    return handle_prediction(predict_murmur_pitch, file)


@app.post("/predict/all/")
async def predict_all_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: Murmur prediction
        murmur_result = predict_murmur(temp_path)
        murmur_class = murmur_result["predicted_class"]

        result = {
            "Murmur": murmur_class,
            "Murmur Timing": "N/A",
            "Murmur Quality": "N/A",
            "Murmur Shape": "N/A",
            "Murmur Pitch": "N/A",
            "Murmur Grade": "N/A"
        }

        # Step 2: If murmur is present, run others
        if murmur_class == "Present":
            result["Murmur Timing"] = predict_murmur_timing(temp_path)["predicted_class"]
            result["Murmur Quality"] = predict_murmur_quality(temp_path)["predicted_class"]
            result["Murmur Shape"] = predict_murmur_shape(temp_path)["predicted_class"]
            result["Murmur Pitch"] = predict_murmur_pitch(temp_path)["predicted_class"]
            result["Murmur Grade"] = predict_murmur_grading(temp_path)["predicted_class"]

        return JSONResponse(content={"Predicted Result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
