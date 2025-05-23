from flask import Flask, request, jsonify 
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import timm

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.path.join('models', 'dense_net.pth')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Criar pasta de uploads se não existir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timm_model = timm.create_model('densenet201', pretrained=True, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNetModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
  
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocessar_imagem(imagem):
    imagem = imagem.convert('L')
    imagem = imagem.resize((96, 96))
    imagem = np.array(imagem, dtype=np.float32) / 255.0
    imagem = np.expand_dims(imagem, axis=0)  # (1, 96, 96)
    imagem = np.expand_dims(imagem, axis=0)  # (1, 1, 96, 96)
    tensor = torch.from_numpy(imagem).to(device)
    return tensor

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API está funcionando!"})

@app.route('/api/detectar', methods=['POST'])
def detectar_pontos():
    try:
        if 'imagem' not in request.files:
            return jsonify({'erro': 'Nenhuma imagem enviada'}), 400
        
        arquivo = request.files['imagem']
        
        if arquivo.filename == '':
            return jsonify({'erro': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_file(arquivo.filename):
            return jsonify({'erro': 'Tipo de arquivo não permitido'}), 400

        imagem = Image.open(io.BytesIO(arquivo.read()))
        tensor = preprocessar_imagem(imagem)

        with torch.no_grad():
            saida = model(tensor)
            pontos = saida.cpu().numpy().flatten().tolist()

        resultado = {
            'pontos': pontos, 
            'dimensoes': imagem.size,
            'mensagem': 'Detecção realizada com sucesso!'
        }
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)