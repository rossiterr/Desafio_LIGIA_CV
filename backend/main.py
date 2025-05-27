from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import timm
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import uvicorn
import cv2
from itertools import pairwise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MobileNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timm_model = timm.create_model('mobilenetv3_large_100', pretrained=True, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

model = MobileNetModel()
model.load_state_dict(torch.load('models/mobile_net.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)),
])

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Warning: Could not load Haar cascade classifier from {face_cascade_path}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents))
    
    image_to_process_pil = image_pil
    face_coordinates = None 

    if not face_cascade.empty():
        image_np = np.array(image_pil.convert('RGB')) 
        print("Image shape:", image_np.shape)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first detected face
            face_coordinates = {"x": x, "y": y, "w": w, "h": h}  # Store face coordinates
            # Crop the original PIL image
            image_to_process_pil = image_pil.crop((x, y, x + w, y + h))
    else:
        pass
    
    image_tensor = transform(image_to_process_pil).unsqueeze(0)

    print("Cropped tensor shape: ", image_tensor.shape)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    keypoints = predictions[0].numpy().tolist()

    def flatten(xss):
        return [x for xs in xss for x in xs]

    # Adjust keypoints back to the original image's coordinate space
    if face_coordinates:
        scale_x = face_coordinates["w"] / 96
        scale_y = face_coordinates["h"] / 96
        adjusted_keypoints = []
        for i in range(0, len(keypoints), 2):
            x, y = keypoints[i], keypoints[i + 1]
            adjusted_keypoints.append([x * scale_x + face_coordinates["x"], y * scale_y + face_coordinates["y"]])
        adjusted_keypoints = flatten(adjusted_keypoints)

    else:
        adjusted_keypoints = keypoints 

    return {"keypoints": adjusted_keypoints}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
