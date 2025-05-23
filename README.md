# Detector de Pontos Faciais

Este projeto é uma aplicação web que permite detectar pontos faciais em imagens usando um modelo de deep learning.

## Estrutura do Projeto

```
projeto/
├── backend/           # API Flask
│   ├── app.py        # Servidor Flask
│   └── requirements.txt
└── frontend/         # Aplicação Next.js
    ├── pages/        # Páginas da aplicação
    └── package.json  # Dependências do frontend
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

2. Instale as dependências:
```bash
cd backend
pip install -r requirements.txt
```

3. Inicie o servidor:
```bash
python app.py
```

O servidor estará rodando em `http://localhost:5000`

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
2. Clique no botão "Upload de Imagem"
3. Selecione uma imagem contendo um rosto
4. Aguarde o processamento
5. Visualize os pontos faciais detectados

## Desenvolvimento

- O backend está em Flask e fornece uma API REST
- O frontend está em Next.js com Material-UI
- O modelo de deep learning será integrado ao backend

## Próximos Passos

- [ ] Integrar o modelo treinado
- [ ] Adicionar mais opções de visualização
- [ ] Implementar detecção em tempo real
- [ ] Adicionar suporte a vídeo
- [ ] Melhorar a interface do usuário 