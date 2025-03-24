"""
Stage 2: Coarse-Grained Action Localization with CLIP

Use the generated action descriptions to derive "actionness" scores for each frame in the video.
"""
import logging
import os
import torch

from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict


def _save_clip_features(
    cfg : DictConfig, 
    features : torch.Tensor,
    video_name : str, 
    output_dir : str,
    type : str,
):
    output_dir = os.path.join(output_dir, type, cfg.stage2.clip_model.arch)
    path = os.path.join(output_dir, f"{video_name}.pt")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    torch.save(features, path)


def _load_clip_features(
    cfg : DictConfig, 
    video_name : str, 
    output_dir : str,
    feat_type : str,
):
    output_dir = os.path.join(output_dir, feat_type, cfg.stage2.clip_model.arch)
    path = os.path.join(output_dir, f"{video_name}.pt")
    
    if os.path.exists(path):
        return torch.load(path, map_location=torch.device('cpu'))
    else:
        return None


def _process_single_video_wclip(
    cfg : DictConfig,
    video_name : str, 
    model_dict : Dict[str, Any],
    output_dir : str, 
    video_reader : Any,
    query_dict : Dict[str, Dict[str, Any]],
):
    ofps = video_reader.get_avg_fps()
    step = max(1, round(ofps / cfg.stage4.fps)) if cfg.stage4.fps > 0 else 1

    num_frames = len(video_reader) // step
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    preprocess = model_dict["preprocess"]

    img_features = _load_clip_features(cfg, video_name.split(".")[0], output_dir, "img_feats")
    img_feats_already_computed = img_features is not None
    logging.info(f"CLIP image features already computed: {img_feats_already_computed}")
    if not img_feats_already_computed:
        img_features = torch.zeros((num_frames, cfg.stage2.clip_model.dim)).half().cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        keys = sorted(query_dict.keys())
        text = tokenizer([query_dict[key]['description'] for key in keys]).cuda()
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for batch_start_idx in tqdm(range(0, num_frames, cfg.stage2.clip_model.batch_size)):
        if not img_feats_already_computed:
            batch_end_idx = min(batch_start_idx + cfg.stage2.clip_model.batch_size, num_frames)
            batch_indices = [i * step for i in range(batch_start_idx, batch_end_idx)]
            batch_frames = video_reader.get_batch(list(batch_indices)).asnumpy()
        
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_images = torch.stack([preprocess(Image.fromarray(frame)) for frame in batch_frames]).cuda()
                batch_image_features = model.encode_image(batch_images)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            img_features[batch_start_idx:batch_end_idx] = batch_image_features

    if not img_feats_already_computed:
        _save_clip_features(cfg, img_features, video_name.split(".")[0], output_dir, "img_feats")

    _save_clip_features(cfg, text_features, video_name.split(".")[0], output_dir, "txt_feats")