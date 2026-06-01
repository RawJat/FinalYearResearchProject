!pip install torch-geometric


# -*- coding: utf-8 -*-
"""Spatially-Aware Image Captioning with Depth and GAT - Revised"""
import os
import json
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.optim as optim

print("all imports successful")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_OBJECTS = 20
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 3
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 5e-5
MAX_LEN = 128

# Paths (Kaggle specific)
IMAGE_DIR = "/kaggle/input/coco-2017-dataset/coco2017/train2017"
ANNOTATION_FILE = "/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json"

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD for GPT-2
vocab_size = tokenizer.vocab_size

class CocoSpatialDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_size=256):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Add resize
            transforms.ToTensor()
        ])
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.image_data = {item['id']: item for item in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann['caption'])
        
        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_data[img_id]['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Select random caption
        caption = random.choice(self.annotations[img_id])
        
        if self.transform:
            image = self.transform(image)
            
        return image, caption, img_id
print("dataset processed")

MIN_OBJECTS = 3

class MaskRCNNDetector:
    def __init__(self, device=device, threshold=0.7):
        self.device = device
        self.model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        self.threshold = threshold
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    @torch.no_grad()
    def detect(self, image):
        # FIX: Handle empty detections properly
        if image.dim() == 4:
            image = image.squeeze(0)
            
        try:
            detections = self.model([image.to(device)])
            detections = detections[0]
            
            # Check for empty detections
            if len(detections['boxes']) == 0:
                return torch.empty(0), [], []
                
            # Filter detections
            keep = detections['scores'] > self.threshold
            boxes = detections['boxes'][keep].cpu()
            
            # Return empty if no detections after thresholding
            if len(boxes) == 0:
                return torch.empty(0), [], []
                
            labels = [self.coco_classes[i] for i in detections['labels'][keep].cpu().numpy()]
            class_indices = detections['labels'][keep].cpu().numpy()
            
            # Limit to top MAX_OBJECTS
            if len(boxes) > MAX_OBJECTS:
                top_indices = np.argsort(detections['scores'][keep].cpu().numpy())[::-1][:MAX_OBJECTS]
                boxes = boxes[top_indices]
                labels = [labels[i] for i in top_indices]
                class_indices = class_indices[top_indices]
            
            return boxes, labels, class_indices
            
        except Exception as e:
            print(f"Detection failed: {str(e)}")
            return torch.empty(0), [], []
print("RCNN done")
class MiDaSDepthEstimator:
    def __init__(self, device=device):
        self.device = device
        self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
        self.model.to(device).eval()
        
        # Create our own transform instead of using MiDaS's
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # MiDaS requires 384x384 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def estimate_depth(self, image):
        # Convert to PIL if necessary
        if isinstance(image, torch.Tensor):
            # Handle batch dimension
            if image.dim() == 4:
                image = image.squeeze(0)
            image = transforms.ToPILImage()(image.cpu())
        elif isinstance(image, np.ndarray):
            # Handle numpy arrays
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure we have a PIL image at this point
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply our custom transform
        input_tensor = self.transform(image).unsqueeze(0).to(device)
        prediction = self.model(input_tensor)
        
        # Resize to original dimensions
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (width, height) -> (height, width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return depth.cpu()
print("Midas done")
class SceneGraphBuilder:
    def __init__(self):
        self.spatial_relations = {
            "left": "left of",
            "right": "right of",
            "above": "above",
            "below": "below",
            "closer": "closer than",
            "farther": "farther than"
        }

    def build_graph(self, boxes, labels, depth_map):
        graph = nx.Graph()
        depth_values = [self.get_object_depth(box, depth_map) for box in boxes]
        
        # Add nodes
        for i, (label, depth_val) in enumerate(zip(labels, depth_values)):
            graph.add_node(i, label=label, depth=depth_val, box=boxes[i])
        
        # Add edges with spatial relationships
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                spatial_rel = self.get_spatial_relationship(
                    boxes[i], boxes[j], depth_values[i], depth_values[j]
                )
                graph.add_edge(i, j, relation=spatial_rel)
        
        return graph

    def get_object_depth(self, box, depth_map):
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
        
        # Check valid region
        if x2 <= x1 or y2 <= y1:
            return torch.tensor(0.0)
        
        depth_roi = depth_map[y1:y2, x1:x2]
        return torch.median(depth_roi).item()

    def get_spatial_relationship(self, box1, box2, depth1, depth2):
        relations = []
        
        # Horizontal relationships
        center_x1 = (box1[0] + box1[2]) / 2
        center_x2 = (box2[0] + box2[2]) / 2
        if center_x1 < center_x2 - 50:  # Add buffer to avoid false positives
            relations.append(self.spatial_relations["left"])
        elif center_x1 > center_x2 + 50:
            relations.append(self.spatial_relations["right"])
        
        # Vertical relationships
        center_y1 = (box1[1] + box1[3]) / 2
        center_y2 = (box2[1] + box2[3]) / 2
        if center_y1 < center_y2 - 50:
            relations.append(self.spatial_relations["above"])
        elif center_y1 > center_y2 + 50:
            relations.append(self.spatial_relations["below"])
        
        # Depth relationships
        if depth1 < depth2:
            relations.append(self.spatial_relations["closer"])
        elif depth1 > depth2:
            relations.append(self.spatial_relations["farther"])
        
        return ", ".join(relations) if relations else "near"

class GATSceneEncoder(nn.Module):
    def __init__(self, num_classes, node_feat_dim=3, hidden_dim=EMBED_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        self.feat_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        # Extract node features
        label_ids = data.x_label
        features = data.x_feat
        
        # Embed label and project features
        label_emb = self.label_embed(label_ids)
        feat_emb = F.relu(self.feat_proj(features))
        x = label_emb + feat_emb
        
        # Process through GAT layers
        x = F.relu(self.gat1(x, data.edge_index))
        x = F.relu(self.gat2(x, data.edge_index))
        return self.norm(x)

class SpatialCaptioningModel(nn.Module):
    def __init__(self, gat_encoder, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.gat_encoder = gat_encoder
        
        # Configure GPT-2 with cross-attention
        config = GPT2Config.from_pretrained("gpt2")
        config.add_cross_attention = True
        config.is_decoder = True
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        self.decoder.resize_token_embeddings(len(tokenizer))
        
        # Projection layers
        self.context_proj = nn.Linear(embed_dim, config.n_embd)
        self.encoder_norm = nn.LayerNorm(config.n_embd)

    def forward(self, graph_data, caption_ids, attention_mask):
        # Encode scene graph
        graph_emb = self.gat_encoder(graph_data)
        graph_emb = torch.mean(graph_emb, dim=0, keepdim=True)  # Global pooling
        
        # Project to decoder space
        encoder_hidden = self.context_proj(graph_emb)
        encoder_hidden = self.encoder_norm(encoder_hidden)
        
        # FIX: Add sequence dimension [batch, seq_len, hidden]
        encoder_hidden = encoder_hidden.unsqueeze(1)  # [1, 1, hidden_size]
        
        # Decode caption
        outputs = self.decoder(
            input_ids=caption_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden,
            labels=caption_ids
        )
        return outputs.loss

    def generate_caption(self, graph_data, max_length=MAX_LEN):
        self.eval()
        with torch.no_grad():
            # Encode scene graph
            graph_emb = self.gat_encoder(graph_data)
            graph_emb = torch.mean(graph_emb, dim=0, keepdim=True)
            
            # Project to decoder space
            encoder_hidden = self.context_proj(graph_emb)
            encoder_hidden = self.encoder_norm(encoder_hidden)
            
            # FIX: Add sequence dimension
            encoder_hidden = encoder_hidden.unsqueeze(1)  # [1, 1, hidden_size]
            
            # Prepare generation
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
            
            for _ in range(max_length):
                outputs = self.decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_hidden
                )
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append token
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def prepare_graph_data(graph, device):
    # FIX: Handle empty graphs
    if graph.number_of_nodes() == 0:
        # Return a dummy graph with one node
        return Data(
            x_feat=torch.zeros((1, 3), dtype=torch.float),
            x_label=torch.zeros(1, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long)
        ).to(device)
    # Node features: [depth, center_x, center_y]
    node_features = []
    label_indices = []
    
    for node in graph.nodes:
        # Get object properties
        box = graph.nodes[node]['box']
        depth = graph.nodes[node]['depth']
        label = graph.nodes[node]['label']
        
        # Calculate center
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Store features
        node_features.append([depth, center_x.item(), center_y.item()])
        label_indices.append(label)  # Store integer class index
    
    # Edge indices (undirected graph)
    edge_indices = []
    for edge in graph.edges:
        edge_indices.append([edge[0], edge[1]])
        edge_indices.append([edge[1], edge[0]])
    
    # Create PyG Data object
    x_feat = torch.tensor(node_features, dtype=torch.float)
    x_label = torch.tensor(label_indices, dtype=torch.long)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(
        x_feat=x_feat, 
        x_label=x_label, 
        edge_index=edge_index
    ).to(device)

def collate_fn(batch):
    images, captions, img_ids = zip(*batch)
    # images = torch.stack(images)
    return images, captions, img_ids

def main():
    # Create dataset and dataloader
    dataset = CocoSpatialDataset(IMAGE_DIR, ANNOTATION_FILE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Initialize models
    detector = MaskRCNNDetector(device)
    depth_estimator = MiDaSDepthEstimator(device)
    graph_builder = SceneGraphBuilder()
    
    # Initialize GAT and captioning model
    gat_encoder = GATSceneEncoder(num_classes=91).to(device)  # 91 COCO classes
    model = SpatialCaptioningModel(gat_encoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        processed_samples = 0
        
        # Use tqdm for progress tracking
        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, captions, img_ids) in progress:
            optimizer.zero_grad()
            batch_loss = 0
            valid_count = 0
            
            for i in range(len(images)):
                try:
                    # Process image through detector
                    boxes, _, class_indices = detector.detect(images[i].unsqueeze(0).to(device))
                    
                    # Skip if not enough objects detected
                    if len(boxes) < MIN_OBJECTS:
                        continue
                    
                    # Estimate depth
                    depth_map = depth_estimator.estimate_depth(images[i])
                    
                    # Build scene graph
                    graph = graph_builder.build_graph(boxes, class_indices, depth_map)
                    
                    # Skip if graph is empty
                    if graph.number_of_nodes() < MIN_OBJECTS:
                        continue
                        
                    graph_data = prepare_graph_data(graph, device)
                    
                    # Tokenize caption
                    inputs = tokenizer(
                        captions[i], 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=MAX_LEN,
                        truncation=True
                    )
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    
                    # Forward pass
                    loss = model(graph_data, input_ids, attention_mask)
                    batch_loss += loss
                    valid_count += 1
                    processed_samples += 1
                    
                except Exception as e:
                    print(f"Error processing image {img_ids[i]}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Skip batch if no valid samples
            if valid_count == 0:
                continue
                
            # Backpropagation
            batch_loss /= valid_count
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            
            # Update progress bar
            progress.set_postfix({"loss": batch_loss.item(), "imgs": processed_samples})
            
            # Clear memory
            torch.cuda.empty_cache()
        
        if processed_samples > 0:
            avg_loss = total_loss / (processed_samples / BATCH_SIZE)
            print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} had no valid samples")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss if processed_samples > 0 else float('inf'),
        }, f"spatial_captioning_epoch_{epoch+1}.pth")
    
    print("Training completed!")
    
    # Inference example
    test_image, test_caption, img_id = dataset[0]
    boxes, _, class_indices = detector.detect(test_image.unsqueeze(0).to(device))
    
    if len(boxes) > 0:
        depth_map = depth_estimator.estimate_depth(test_image)
        graph = graph_builder.build_graph(boxes, class_indices, depth_map)
        graph_data = prepare_graph_data(graph, device)
        
        generated_caption = model.generate_caption(graph_data)
        print("\nExample Caption Generation:")
        print(f"Original Caption: {test_caption}")
        print(f"Generated Caption: {generated_caption}")
    else:
        print("No objects detected in test image")

if __name__ == "__main__":
    from tqdm import tqdm
    main()

import requests
from io import BytesIO
import torch
from PIL import Image

# Load your trained model
def load_model(checkpoint_path):
    # Initialize components (same as training)
    gat_encoder = GATSceneEncoder(num_classes=91).to(device)
    model = SpatialCaptioningModel(gat_encoder).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, MaskRCNNDetector(device), MiDaSDepthEstimator(device)

# Generate caption from URL
def caption_from_url(url, model, detector, depth_estimator):
    # Download image
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Preprocess (same as dataset)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Process through pipeline
    boxes, _, class_indices = detector.detect(img_tensor)
    depth_map = depth_estimator.estimate_depth(img)
    graph = SceneGraphBuilder().build_graph(boxes, class_indices, depth_map)
    graph_data = prepare_graph_data(graph, device)
    
    # Generate caption
    caption = model.generate_caption(graph_data)
    return caption

# Usage:
MODEL_PATH = '/kaggle/working/spatial_captioning_epoch_{epoch+1}.pth'
model, detector, depth_estimator = load_model(MODEL_PATH)

url = "https://www.myfooddiary.com/blog/asset/1999/tips_starting_walking_group.jpg"
caption = caption_from_url(url, model, detector, depth_estimator)
print(f"Generated Caption: {caption}")

!pip install transformers accelerate bitsandbytes

from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration
import torch

# Load lightweight BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16,
    device_map="auto"
)

# Initialize conversation
conversation_history = ""

def chat_with_image(url, question):
    global conversation_history
    
    # Download image
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Format prompt
    prompt = f"{conversation_history}Question: {question} Answer:"
    
    # Generate response
    inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=100)
    answer = processor.decode(out[0], skip_special_tokens=True).strip()
    
    # Update history
    conversation_history += f"Question: {question} Answer: {answer} "
    return answer

# Example conversation
url = "https://www.myfooddiary.com/blog/asset/1999/tips_starting_walking_group.jpg"

print(chat_with_image(url, "What's in this image?"))
print(chat_with_image(url, "Where is the cat positioned?"))
print(chat_with_image(url, "What's to the left of the dog?"))

