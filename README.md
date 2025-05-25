# Detector de Pontos Faciais

Este projeto é uma aplicação web que permite detectar pontos faciais em imagens usando um modelo de deep learning.

## Estrutura do Projeto

```
projeto/
├── backend/           # Fast API
│   ├── main.py        # Script do Servidor
│   └── requirements.txt
└── frontend/          # Aplicação em react
    ├── src/           # Pasta fonte do front end
    │   └── App.tsx    # Página da aplicação
    └── package.json   # Dependências do frontend
```

## Requisitos

- Python 3.8+
- Node.js 14+
- npm ou yarn

## Instalação

### Backend

1. Crie um ambiente virtual Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Instale as dependências no venv:
```bash
cd backend
python -m pip install --upgrade pip
pip install setuptools wheel       
pip install -r requirements.txt
```

3. Inicie o servidor:
```bash
python main.py
```

O servidor estará rodando em `http://localhost:8000`

### Frontend

1. Instale as dependências:
```bash
cd frontend
npm install
```

2. Inicie o servidor de desenvolvimento:
```bash
npm run dev
```

A aplicação estará disponível em `http://localhost:3000`

## Uso

1. Acesse a aplicação em `http://localhost:3000`
2. Permita acesso da webcam à aplicação
3. Tire uma foto pela webcam utilizando o botão
4. Selecione a opção de detectar pontos faciais
5. Aguarde o processamento
6. Visualize os pontos faciais detectados

## Desenvolvimento

- O backend está em utilizando FastAPI
- O frontend está em React com Material-UI
- O modelo de deep learning deve ser integrado manualmente a pasta models no backend
