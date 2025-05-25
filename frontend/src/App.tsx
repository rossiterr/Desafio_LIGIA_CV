import React, { useState, useRef } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Paper,
  CircularProgress
} from '@mui/material';
import axios from 'axios';

interface Keypoint {
  x: number;
  y: number;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        setKeypoints([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append('file', blob, 'image.jpg');

      const result = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Convertendo as coordenadas para o formato Keypoint
      const points = result.data.keypoints;
      const keypointsArray: Keypoint[] = [];
      for (let i = 0; i < points.length; i += 2) {
        keypointsArray.push({
          x: points[i],
          y: points[i + 1]
        });
      }
      setKeypoints(keypointsArray);
    } catch (error) {
      console.error('Erro ao fazer a predição:', error);
    } finally {
      setLoading(false);
    }
  };

  const drawImage = (ctx: CanvasRenderingContext2D, image: HTMLImageElement) => {
    ctx.drawImage(image, 0, 0, 400, 400);
  };

  const drawKeypoints = (ctx: CanvasRenderingContext2D, image: HTMLImageElement) => {
    drawImage(ctx, image);
    
    // Desenhando os pontos faciais
    ctx.fillStyle = 'red';
    keypoints.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x * 4, point.y * 4, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  React.useEffect(() => {
    if (selectedImage) {
      const canvas = document.getElementById('canvas') as HTMLCanvasElement;
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
      
      image.src = selectedImage;
    }
  }, [selectedImage, keypoints]);

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Detecção de Pontos Faciais
        </Typography>
        
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={{ display: 'none' }}
            ref={fileInputRef}
          />
          
          <Button
            variant="contained"
            onClick={() => fileInputRef.current?.click()}
            sx={{ mr: 2 }}
          >
            Selecionar Imagem
          </Button>
          
          <Button
            variant="contained"
            onClick={handlePredict}
            disabled={!selectedImage || loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Detectar Pontos'}
          </Button>
        </Paper>

        {selectedImage && (
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <canvas
              id="canvas"
              width={400}
              height={400}
              style={{ border: '1px solid #ccc' }}
            />
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default App;
