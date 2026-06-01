!pip install 'git+https://github.com/facebookresearch/detectron2.git'




# Spatially Grounded Image Captioning Pipeline (Training to Inference)

import torch
import torch.nn as nn
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, CLIPProcessor, CLIPModel
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from transformers import AutoTokenizer, AutoModelForCausalLM
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Image Loader
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# 2. Setup Mask R-CNN
def setup_mask_rcnn():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return predictor, class_names

# 3. Setup MiDaS
def setup_depth_model():
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    return model, processor

def get_depth_map(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return cv2.resize(depth, image.size)

# 4. Compute Spatial Scene Facts
def compute_scene_facts(objects, masks, depth_map):
    facts = []
    for (i, obj_a), (j, obj_b) in itertools.combinations(enumerate(objects), 2):
        label_a, label_b = obj_a['label'], obj_b['label']
        mask_a, mask_b = masks[i], masks[j]

        contact_pixels = np.logical_and(mask_a, mask_b).sum()
        contact = contact_pixels > 30

        coords_a = np.argwhere(mask_a)
        coords_b = np.argwhere(mask_b)
        center_a = coords_a.mean(axis=0)
        center_b = coords_b.mean(axis=0)

        dx, dy = center_b[1] - center_a[1], center_b[0] - center_a[0]
        direction = ("right" if dx > abs(dy) else
                     "left" if dx < -abs(dy) else
                     "below" if dy > 0 else "above")

        def avg_depth(mask): return depth_map[mask].mean()
        depth_a, depth_b = avg_depth(mask_a), avg_depth(mask_b)
        if abs(depth_a - depth_b) > 0.05:
            closer = label_a if depth_a < depth_b else label_b
            farther = label_b if depth_a < depth_b else label_a
            facts.append(f"The {closer} is closer than the {farther}.")

        if contact:
            facts.append(f"The {label_a} is in contact with the {label_b}.")
        facts.append(f"The {label_a} is {direction} of the {label_b}.")
    return facts

# 5. Vision Encoder (ViT-L/14 via CLIP)
def get_image_embedding(image):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_embeds = model.get_image_features(**inputs)
    return vision_embeds

# 6. Language Model (LLaMA or Vicuna)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-7B-1.1-HF")
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/vicuna-7B-1.1-HF",
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    return tokenizer, model


def generate_caption(scene_facts, vision_embedding, tokenizer, model):
    prompt = (
        "### System: You are a visual assistant that strictly describes a scene based only on the provided facts.\n"
        "You are not allowed to make assumptions or inferences on contact. Only refer to contact, proximity, or object relationships if they are explicitly stated in the facts.\n"
        "If something is not in the facts, do not mention it.\n\n"
        "### User: Describe the image in a concise paragraph.\n"
        "### Scene Facts:\n" + "\n".join(f"- {fact}" for fact in scene_facts) + "\n"
        "### Assistant:"
    )



    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        output = model.generate(inputs, max_new_tokens=100)
    caption = tokenizer.decode(output[0], skip_special_tokens=True).split("### Assistant:")[-1].strip()
    return caption

# 7. Full Pipeline
def run_full_pipeline(image_url):
    print("[INFO] Loading image...")
    image = load_image_from_url(image_url)

    print("[INFO] Setting up Mask R-CNN...")
    predictor, class_names = setup_mask_rcnn()

    print("[INFO] Setting up MiDaS depth model...")
    depth_model, depth_processor = setup_depth_model()

    print("[INFO] Running segmentation...")
    img_np = np.array(image)[:, :, ::-1].copy()
    outputs = predictor(img_np)
    instances = outputs["instances"]
    masks = instances.pred_masks.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    from collections import defaultdict
    counter = defaultdict(int)
    objects = []
    for i, cls_id in enumerate(classes):
        label = class_names[cls_id]
        counter[label] += 1
        objects.append({"label": f"{label}_{counter[label]}"})

    print("[INFO] Computing depth map...")
    depth_map = get_depth_map(image, depth_model, depth_processor)

    print("[INFO] Computing scene facts...")
    scene_facts = compute_scene_facts(objects, masks, depth_map)
    print("[DEBUG] Scene facts:", scene_facts)

    print("[INFO] Getting vision embeddings...")
    image_embedding = get_image_embedding(image)

    print("[INFO] Loading LLM...")
    tokenizer, llm = load_llm()

    print("[INFO] Generating caption...")
    caption = generate_caption(scene_facts, image_embedding, tokenizer, llm)

    print("\n✅ Generated Caption:\n", caption)
    return caption, scene_facts, tokenizer, llm



image_url = "https://miro.medium.com/v2/resize:fit:640/format:webp/1*C55KVBoSugntG0MLkmAx7A.jpeg"
run_full_pipeline(image_url)


def chat_with_model(user_input, scene_facts=None, tokenizer=None, model=None):
    if scene_facts:
        context = "\n".join(f"- {fact}" for fact in scene_facts)
        prompt = (
            "### System: You are a visual assistant that describes scenes and answers questions **only** based on the provided scene facts. "
            "You must not guess or infer contact, positions, or relationships unless they are **explicitly mentioned** in the facts. "
            "If the answer is not in the facts, respond with: 'The facts do not provide enough information to answer that.'\n\n"
            f"### Scene Facts:\n{context}\n"
            f"### User: {user_input}\n"
            "### Assistant:"
        )

    else:
        prompt = (
            "### System: You are a helpful assistant.\n"
            f"### User: {user_input}\n"
            "### Assistant:"
        )

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        output = model.generate(inputs, max_new_tokens=150)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("### Assistant:")[-1].strip()
    return response



caption, scene_facts, tokenizer, model = run_full_pipeline(image_url)




while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = chat_with_model(user_input, scene_facts, tokenizer, model)
    print("AI:", response)








