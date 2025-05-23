import { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Container, 
  Typography, 
  CircularProgress,
  Alert,
  Paper,
  Tabs,
  Tab
} from '@mui/material';
import { styled } from '@mui/material/styles';
import axios from 'axios';

const CANVAS_SIZE = 400;

const VisorImagem = styled('canvas')({
  width: `${CANVAS_SIZE}px`,
  height: `${CANVAS_SIZE}px`,
  maxWidth: `${CANVAS_SIZE}px`,
  maxHeight: `${CANVAS_SIZE}px`,
  border: '2px dashed #ccc',
  borderRadius: '8px',
  marginTop: '20px',
});

const UploadBox = styled(Paper)({
  padding: '20px',
  textAlign: 'center',
  marginTop: '20px',
  backgroundColor: '#f5f5f5',
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

export default function Home() {
  const [imagem, setImagem] = useState<File | null>(null);
  const [pontos, setPontos] = useState<number[]>([]);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const desenharPontos = (imagem: HTMLImageElement, pontos: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Define tamanho fixo do canvas
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Desenha a imagem redimensionada para o tamanho do canvas
    ctx.drawImage(imagem, 0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctx.fillStyle = 'red';
    for (let i = 0; i < pontos.length; i += 2) {
      // Escala os pontos do range [0, 96] para o tamanho do canvas
      const x = (pontos[i] / 96) * CANVAS_SIZE;
      const y = (pontos[i + 1] / 96) * CANVAS_SIZE;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const processarImagem = async (arquivo: File) => {
    setImagem(arquivo);
    setErro(null);
    setCarregando(true);

    const url = URL.createObjectURL(arquivo);
    setPreviewUrl(url);

    const formData = new FormData();
    formData.append('imagem', arquivo);

    try {
      const resposta = await axios.post('http://localhost:5000/api/detectar', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPontos(resposta.data.pontos);

      const img = new Image();
      img.onload = () => desenharPontos(img, resposta.data.pontos);
      img.src = url;

    } catch (erro) {
      console.error('Erro:', erro);
      setErro('Erro ao processar a imagem. Tente novamente.');
    } finally {
      setCarregando(false);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const arquivo = event.target.files?.[0];
    if (!arquivo) return;
    processarImagem(arquivo);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Detector de Pontos Faciais
        </Typography>

        <Typography variant="body1" color="text.secondary" paragraph>
          Faça upload de uma imagem para detectar os pontos faciais
        </Typography>

        <UploadBox elevation={3}>
          <Button
            variant="contained"
            component="label"
            disabled={carregando}
            sx={{ my: 2 }}
          >
            {carregando ? 'Processando...' : 'Upload de Imagem'}
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={handleUpload}
            />
          </Button>

          {carregando && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress />
            </Box>
          )}

          {erro && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {erro}
            </Alert>
          )}
        </UploadBox>

        {previewUrl && (
          <Box sx={{ mt: 2 }}>
            <VisorImagem
              ref={canvasRef}
            />
          </Box>
        )}
      </Box>
    </Container>
  );
}