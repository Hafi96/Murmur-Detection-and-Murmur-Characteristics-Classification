# app/predicion.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from .HMSSNet import Hierachical_MS_Net
from .data import Preprocessor

#1 murmur predcition
CLASSES = ["Present", "Absent"]
def predict_murmur(audio_path: str, model_path: str = "models/murmur_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(CLASSES, np.round(probs, 3).tolist()))
    }
    
#2 murmur timing prediction
Timing_CLASSES = ["Holosystolic", "Early-systolic"]
def predict_murmur_timing(audio_path: str, model_path: str = "models/timing_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(Timing_CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = Timing_CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(Timing_CLASSES, np.round(probs, 3).tolist()))
    } 

#3 murmur grading prediction
grading_CLASSES = ["I/VI","II/VI","III/VI"]
def predict_murmur_grading(audio_path: str, model_path: str = "models/grade_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(grading_CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = grading_CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(grading_CLASSES, np.round(probs, 3).tolist()))
    } 
    
#4 murmur quality prediction
quality_CLASSES = ['Blowing','Harsh']
def predict_murmur_quality(audio_path: str, model_path: str = "models/quality_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(quality_CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = quality_CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(quality_CLASSES, np.round(probs, 3).tolist()))
    }
#5 murmur shape prediction
shape_CLASSES = ["Diamond","Plateau","Decrescendo"]
def predict_murmur_shape(audio_path: str, model_path: str = "models/shape_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(shape_CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = shape_CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(shape_CLASSES, np.round(probs, 3).tolist()))
    }
#6 murmur pitch prediction
pitch_CLASSES = ["Low","Medium","High"]
def predict_murmur_pitch(audio_path: str, model_path: str = "models/pitch_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hierachical_MS_Net(num_classes=len(pitch_CLASSES), include_patient_data=False)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs, signal = wavfile.read(audio_path)
    preprocessor = Preprocessor(frequency=2000, mode='test')
    multi_scale_specs, _ = preprocessor(signal, fs)
    multi_scale_specs = [s.to(device) for s in multi_scale_specs]

    with torch.no_grad():
        output = model(multi_scale_specs)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probs))
        predicted_class = pitch_CLASSES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(pitch_CLASSES, np.round(probs, 3).tolist()))
    }