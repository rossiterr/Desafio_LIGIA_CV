# Detecção de Pontos Faciais

Este projeto é uma aplicação web que utiliza um modelo de deep learning para detectar pontos faciais em imagens. A aplicação consiste em um frontend em React com TypeScript e um backend em Python usando FastAPI.

## Estrutura do Projeto

```
.
├── frontend/           # Aplicação React
├── backend/           # API FastAPI
│   └── models/        # Modelos treinados
└── AI/               # Notebooks e código de treinamento
```

## Requisitos

### Frontend
- Node.js 14+
- npm 6+

### Backend
- Python 3.8+
- PyTorch
- FastAPI
- Outras dependências listadas em `backend/requirements.txt`

## Instalação

### Frontend

1. Entre na pasta do frontend:
```bash
cd frontend
```

2. Instale as dependências:
```bash
npm install
```

3. Inicie o servidor de desenvolvimento:
```bash
npm start
```

O frontend estará disponível em `http://localhost:3000`

### Backend

1. Entre na pasta do backend:
```bash
cd backend
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Inicie o servidor:
```bash
python main.py
```

O backend estará disponível em `http://localhost:8000`

## Uso

1. Acesse a aplicação em `http://localhost:3000`
2. Clique em "Selecionar Imagem" para fazer upload de uma imagem
3. Clique em "Detectar Pontos" para processar a imagem
4. Os pontos faciais serão exibidos sobrepostos à imagem

## API Endpoints

### POST /predict
Recebe uma imagem e retorna as coordenadas dos pontos faciais detectados.

**Request:**
- Content-Type: multipart/form-data
- Body: file (imagem)

**Response:**
```json
{
  "keypoints": [x1, y1, x2, y2, ...]  // 30 valores (15 pares de coordenadas)
}
```

## Modelo

O modelo utilizado é baseado em MobileNetV3, treinado para detectar 30 pontos faciais (15 pares de coordenadas x,y). O modelo recebe imagens em escala de cinza (1 canal) e retorna as coordenadas dos pontos faciais.

## Tecnologias Utilizadas

- Frontend:
  - React
  - TypeScript
  - Material-UI
  - Axios

- Backend:
  - Flask
  - PyTorch
  - EfficientNetV2
  - OpenCV

## Desenvolvimento

- O backend está em Flask e fornece uma API REST
- O frontend está em React com Material-UI
- O modelo de deep learning será integrado ao backend

## Próximos Passos

- [ ] Integrar o modelo treinado
- [ ] Adicionar mais opções de visualização
- [ ] Implementar detecção em tempo real
- [ ] Adicionar suporte a vídeo
- [ ] Melhorar a interface do usuário 