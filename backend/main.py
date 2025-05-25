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

app = FastAPI()

# Configurando CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definindo o modelo
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

# Carregando o modelo
model = MobileNetModel()
model.load_state_dict(torch.load('models/mobile_net.pth', map_location=torch.device('cpu')))
model.eval()

# Transformações para a imagem
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lendo a imagem
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Aplicando transformações
    image_tensor = transform(image).unsqueeze(0)
    
    # Fazendo a predição
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Convertendo as predições para lista
    keypoints = predictions[0].numpy().tolist()
    
    # Retornando as coordenadas dos pontos faciais
    return {"keypoints": keypoints}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 