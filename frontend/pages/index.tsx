import { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Container, 
  Typography, 
  CircularProgress,
  Alert,
  Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import axios from 'axios';

const VisorImagem = styled('canvas')({
  maxWidth: '100%',
  height: 'auto',
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

export default function Home() {
  const [imagem, setImagem] = useState<File | null>(null);
  const [pontos, setPontos] = useState<number[]>([]);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    console.log(pontos);
  }, [pontos]);

  const desenharPontos = (imagem: HTMLImageElement, pontos: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Configurar o canvas com as dimensões da imagem
    canvas.width = imagem.width;
    canvas.height = imagem.height;

    // Desenhar a imagem
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imagem, 0, 0, canvas.width, canvas.height);
    
    // Desenhar os pontos
    ctx.fillStyle = 'red';
    for (let i = 0; i < pontos.length; i += 2) {
      const x = pontos[i] * canvas.width;
      const y = pontos[i + 1] * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const processarImagem = async (arquivo: File) => {
    setImagem(arquivo);
    setErro(null);
    setCarregando(true);

    try {
      // Criar URL para preview
      const url = URL.createObjectURL(arquivo);
      setPreviewUrl(url);

      // Preparar e enviar para a API
      const formData = new FormData();
      formData.append('imagem', arquivo);

      const resposta = await axios.post('http://localhost:5000/api/detectar', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Atualizar pontos e desenhar na imagem
      setPontos(resposta.data.pontos);

      const img = new Image();
      img.onload = () => {
        desenharPontos(img, resposta.data.pontos);
        // Limpar a URL do objeto após carregar a imagem
        URL.revokeObjectURL(url);
      };
      img.src = url;

    } catch (err) {
      console.error('Erro:', err);
      setErro('Erro ao processar a imagem. Tente novamente.');
      // Limpar preview em caso de erro
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
    } finally {
      setCarregando(false);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const arquivo = event.target.files?.[0];
    if (!arquivo) return;
    processarImagem(arquivo);
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
              width={500}
              height={500}
            />
          </Box>
        )}
      </Box>
    </Container>
  );
} 