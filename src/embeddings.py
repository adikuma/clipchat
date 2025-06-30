import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import BridgeTowerProcessor, BridgeTowerModel
from rich.console import Console
from rich.progress import Progress

console = Console()

class EmbeddingGenerator:
    """generates multimodal embeddings using bridgetower model"""
    
    def __init__(self, model_name: str = "BridgeTower/bridgetower-base"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[blue]using device: {self.device}[/blue]")
    
    def load_model(self):
        """load bridgetower model and processor"""
        if self.processor is None or self.model is None:
            console.print(f"[blue]loading {self.model_name}...[/blue]")
            
            self.processor = BridgeTowerProcessor.from_pretrained(self.model_name)
            self.model = BridgeTowerModel.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            console.print("[green]model loaded successfully[/green]")
    
    def create_dummy_image(self, size: tuple = (224, 224)) -> Image.Image:
        """create dummy white image for text-only embeddings"""
        return Image.new("RGB", size, (255, 255, 255))
    
    def generate_embedding(self, image: Image.Image, text: str) -> np.ndarray:
        """
        generate single multimodal embedding
        
        args:
            image: pil image
            text: text string
            
        returns:
            numpy array of embedding
        """
        self.load_model()
        
        # prepare inputs
        inputs = self.processor(image, text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # return pooled output
        embedding = outputs.pooler_output.squeeze().cpu().numpy()
        return embedding
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        generate embedding for text using dummy image
        
        args:
            text: text string
            
        returns:
            numpy array of embedding
        """
        dummy_image = self.create_dummy_image()
        return self.generate_embedding(dummy_image, text)
    
    def generate_batch_embeddings(
        self, 
        image_paths: List[str], 
        texts: List[str],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        generate embeddings for multiple image-text pairs
        
        args:
            image_paths: list of image file paths
            texts: list of text strings
            batch_size: number of items to process at once
            
        returns:
            list of numpy arrays (embeddings)
        """
        self.load_model()
        
        if len(image_paths) != len(texts):
            raise ValueError("number of images and texts must match")
        
        embeddings = []
        
        with Progress() as progress:
            task = progress.add_task("generating embeddings...", total=len(image_paths))
            
            for i in range(0, len(image_paths), batch_size):
                batch_images = []
                batch_texts = []
                
                # prepare batch
                for j in range(i, min(i + batch_size, len(image_paths))):
                    try:
                        img = Image.open(image_paths[j]).convert("RGB")
                        batch_images.append(img)
                        batch_texts.append(texts[j])
                    except Exception as e:
                        console.print(f"[red]error loading {image_paths[j]}: {e}[/red]")
                        # use dummy image for failed loads
                        batch_images.append(self.create_dummy_image())
                        batch_texts.append(texts[j])
                
                # process batch
                try:
                    # process each item in batch individually for now
                    # (bridgetower processor may not support batching properly)
                    for img, text in zip(batch_images, batch_texts):
                        embedding = self.generate_embedding(img, text)
                        embeddings.append(embedding)
                        progress.advance(task)
                        
                except Exception as e:
                    console.print(f"[red]batch processing error: {e}[/red]")
                    # fallback to individual processing
                    for img, text in zip(batch_images, batch_texts):
                        try:
                            embedding = self.generate_embedding(img, text)
                            embeddings.append(embedding)
                        except Exception as e2:
                            console.print(f"[red]individual embedding error: {e2}[/red]")
                            # create zero embedding as fallback
                            embedding = np.zeros(768)  # bridgetower base output dim
                            embeddings.append(embedding)
                        progress.advance(task)
        
        console.print(f"[green]generated {len(embeddings)} embeddings[/green]")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """get the dimension of embeddings produced by the model"""
        self.load_model()
        
        # test with dummy data
        dummy_img = self.create_dummy_image()
        test_embedding = self.generate_embedding(dummy_img, "test")
        
        return len(test_embedding)