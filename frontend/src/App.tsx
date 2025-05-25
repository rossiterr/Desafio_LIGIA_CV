import { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Paper,
  CircularProgress,
  Alert,
  Stack,
} from '@mui/material';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import axios from 'axios';

interface Keypoint {
  x: number;
  y: number;
}

const CANVAS_WIDTH = 400;
const CANVAS_HEIGHT = 400;

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isWebcamStarted, setIsWebcamStarted] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null); // Ref para a tela de exibição

  // Função para iniciar a webcam
  const startWebcam = useCallback(async () => {
    if (videoRef.current) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        setIsWebcamStarted(true);
        setError(null); // Limpar erros anteriores
      } catch (err) {
        console.error('Erro ao aceder à webcam:', err);
        setError('Não foi possível aceder à webcam. Verifique as permissões e tente novamente.');
        setIsWebcamStarted(false);
      }
    }
  }, []);

  // Função para parar a webcam
  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsWebcamStarted(false);
    }
  }, []);

  // Efeito para gerir o estado da webcam
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  const handleCapture = () => {
    const video = videoRef.current;
    const tempCanvas = document.createElement('canvas'); 
    if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
      tempCanvas.width = CANVAS_WIDTH;
      tempCanvas.height = CANVAS_HEIGHT;
      const ctx = tempCanvas.getContext('2d');
      if (ctx) {
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const videoAspectRatio = videoWidth / videoHeight;
        const canvasAspectRatio = CANVAS_WIDTH / CANVAS_HEIGHT;
        let drawWidth, drawHeight, offsetX, offsetY;

        if (videoAspectRatio > canvasAspectRatio) { 
            drawHeight = CANVAS_HEIGHT;
            drawWidth = drawHeight * videoAspectRatio;
            offsetX = (CANVAS_WIDTH - drawWidth) / 2;
            offsetY = 0;
        } else { 
            drawWidth = CANVAS_WIDTH;
            drawHeight = drawWidth / videoAspectRatio;
            offsetY = (CANVAS_HEIGHT - drawHeight) / 2;
            offsetX = 0;
        }
        
        ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
        const dataUrl = tempCanvas.toDataURL('image/jpeg');
        setSelectedImage(dataUrl);
        setKeypoints([]); 
        setError(null);
      }
    } else {
      setError("Webcam não pronta ou nenhum stream de vídeo disponível para captura.");
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append('file', blob, 'image.jpg');

      // IMPORTANTE: Substitua pelo seu endpoint de API real
      const result = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const points = result.data.keypoints;
      if (!points || points.length % 2 !== 0) {
        console.error('Dados de keypoints inválidos:', points);
        setError('Recebidos dados de keypoints inválidos do servidor.');
        setKeypoints([]);
        return;
      }
      
      const keypointsArray: Keypoint[] = [];
      for (let i = 0; i < points.length; i += 2) {
        keypointsArray.push({
          x: points[i],
          y: points[i + 1]
        });
      }
      setKeypoints(keypointsArray);
    } catch (err) {
      console.error('Erro durante a predição:', err);
      setError('Falha ao obter a predição. O servidor pode estar em baixo ou ocorreu um erro.');
      setKeypoints([]); 
    } finally {
      setLoading(false);
    }
  };

  const drawImage = useCallback((ctx: CanvasRenderingContext2D, image: HTMLImageElement) => {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT); 
    ctx.drawImage(image, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }, []);

  const drawKeypoints = useCallback((ctx: CanvasRenderingContext2D, image: HTMLImageElement) => {
    drawImage(ctx, image); 
    
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)'; 
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'; 
    ctx.lineWidth = 1;

    keypoints.forEach((point) => {
      const kx = point.x * (CANVAS_WIDTH / 100); 
      const ky = point.y * (CANVAS_HEIGHT / 100);

      ctx.beginPath();
      ctx.arc(kx, ky, 4, 0, 2 * Math.PI); 
      ctx.fill();
      ctx.stroke();
    });
  }, [keypoints, drawImage]);

  useEffect(() => {
    if (selectedImage && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const image = new Image();
      
      image.onload = () => {
        if (ctx) {
          if (keypoints.length > 0) {
            drawKeypoints(ctx, image);
          } else {
            drawImage(ctx, image);
          }
        }
      };
      image.onerror = () => {
        setError("Falha ao carregar a imagem capturada para exibição.");
      }
      image.src = selectedImage;
    } else if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        ctx.fillStyle = '#f0f0f0'; 
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        ctx.fillStyle = '#777';
        ctx.textAlign = 'center';
        ctx.fillText('A imagem capturada aparecerá aqui', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
      }
    }
  }, [selectedImage, keypoints, drawImage, drawKeypoints]);

  const handleReset = () => {
    setSelectedImage(null);
    setKeypoints([]);
    setError(null);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: { xs: 2, md: 4 }, borderRadius: 3 }}>
        <Typography variant="h3" component="h1" gutterBottom textAlign="center" fontWeight="bold" color="primary.main">
          Deteção de Pontos Chave Faciais
        </Typography>
        <Typography variant="subtitle1" textAlign="center" color="text.secondary" mb={4}>
          Use a sua webcam para capturar uma imagem e detetar os pontos chave faciais.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Usando Box com Flexbox para layout de coluna */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' }, 
            alignItems: 'stretch', 
            gap: theme => theme.spacing(4), 
          }}
        >
          {/* Secção da Webcam */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%' }}>
              <Typography variant="h6" gutterBottom component="h2">Feed da Webcam</Typography>
              <Box sx={{ 
                width: '100%', 
                aspectRatio: '1/1', 
                backgroundColor: '#e0e0e0', 
                border: '2px dashed #bdbdbd',
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden', 
                position: 'relative'
              }}>
                <video
                  ref={videoRef}
                  width={CANVAS_WIDTH} 
                  height={CANVAS_HEIGHT} 
                  autoPlay
                  playsInline 
                  muted 
                  style={{ 
                    display: isWebcamStarted ? 'block' : 'none',
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover', 
                    borderRadius: 'inherit'
                  }}
                />
                {!isWebcamStarted && (
                    <Typography variant="caption" color="textSecondary" sx={{position: 'absolute'}}>
                        Webcam desligada ou não iniciada.
                    </Typography>
                )}
              </Box>
              <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                {!isWebcamStarted ? (
                    <Button
                        variant="contained"
                        color="secondary"
                        startIcon={<VideocamIcon />}
                        onClick={startWebcam}
                    >
                        Iniciar Webcam
                    </Button>
                ) : (
                    <Button
                        variant="outlined"
                        color="secondary"
                        startIcon={<VideocamOffIcon />}
                        onClick={stopWebcam}
                    >
                        Parar Webcam
                    </Button>
                )}
                <Button
                  variant="contained"
                  startIcon={<PhotoCameraIcon />}
                  onClick={handleCapture}
                  disabled={!isWebcamStarted || loading}
                >
                  Capturar Foto
                </Button>
              </Stack>
            </Paper>
          </Box>

          {/* Secção da Imagem Capturada e Predição */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%' }}>
              <Typography variant="h6" gutterBottom component="h2">Imagem Capturada e Pontos Chave</Typography>
              <Box sx={{ 
                  width: '100%', 
                  aspectRatio: '1/1', 
                  border: '2px solid #ccc', 
                  borderRadius: 2,
                  backgroundColor: selectedImage ? 'transparent' : '#f0f0f0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden' 
              }}>
                <canvas
                  ref={canvasRef}
                  width={CANVAS_WIDTH} 
                  height={CANVAS_HEIGHT} 
                  style={{ 
                    borderRadius: 'inherit',
                    width: '100%', 
                    height: '100%'  
                  }}
                />
              </Box>
              <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<PlayArrowIcon />}
                  onClick={handlePredict}
                  disabled={!selectedImage || loading}
                  sx={{ minWidth: '160px' }}
                >
                  {loading ? <CircularProgress size={24} color="inherit" /> : 'Detetar Pontos'}
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<RestartAltIcon />}
                  onClick={handleReset}
                  disabled={loading}
                >
                  Repor
                </Button>
              </Stack>
            </Paper>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}

export default App;
