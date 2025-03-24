"""
Stage 4: Fine-Grained Action Localization with VLM Confidence

Compute frame-level confidence scores using VLM output yes/no token logits for each action query.
"""
import copy
import logging
import numpy as np
import os
import sys
import textwrap
import torch

from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Any, Dict

from stage3_filter_action_classes import _path


def make_vlm_query(
    cfg : DictConfig,
    subquery: str,
    action_class: str,
):
    if cfg.score_strategy == "SEQ":
        return textwrap.dedent(f"""
            This is a frame from a video of {action_class}.
            {subquery} Only answer yes or no.
        """)
    elif cfg.score_strategy == "PHR":
        return textwrap.dedent(f"""
            This is a frame from a video of {action_class}.
            Describe the actions in this frame in a short phrase: {subquery}
        """)
    elif cfg.score_strategy == "PHR-SEQ":
        raise NotImplementedError


def _save_scores(
    cfg : DictConfig, 
    scores : np.ndarray, 
    video_name : str, 
    output_dir : str,
):
    path = os.path.join(
        output_dir, 
        f"{video_name}_vlm_scores_{cfg.score_strategy}{cfg.stage4.fps}_{cfg.stage1.llm_type}{_path(cfg)}"
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(path, scores)


def _load_scores(
    cfg : DictConfig, 
    video_name : str, 
    output_dir : str,
):
    path = os.path.join(
        output_dir, 
        f"{video_name}_vlm_scores_{cfg.score_strategy}{cfg.stage4.fps}_{cfg.stage1.llm_type}{_path(cfg)}"
    )

    return np.load(path) if os.path.exists(path) else None


def _process_single_video_wvlm(
    cfg : DictConfig,
    video_name : str, 
    model_dict : Dict[str, Any],
    output_dir : str, 
    video_reader : Any,
    query_dict : Dict[str, Dict[str, Any]],
    label_map : Dict[str, int],
):
    scores = _load_scores(cfg, video_name.split(".")[0], output_dir)

    if scores is not None and cfg.stage4.overwrite == False:
        logging.warning(
            f"VLM soft-scores for {video_name} already computed. Skipping..."
        )
    else:
        ofps = video_reader.get_avg_fps()
        step = max(1, round(ofps / cfg.stage4.fps)) if cfg.stage4.fps > 0 else 1

        num_frames = len(video_reader)
        num_classes = cfg.dataset.num_classes
        num_steps = (num_frames + step - 1) // step
        scores = np.zeros((num_steps, num_classes, 2))

        for i, frame_idx in tqdm(enumerate(range(0, num_frames, step)), total=num_steps):
            frame = video_reader[frame_idx].asnumpy()
            frame = Image.fromarray(frame)

            for j, (action_class, query) in enumerate(query_dict.items()):
                for k, (qkind, subquery) in enumerate(query.items()):
                    if qkind == "description":
                        continue

                    k = 0 if qkind == "start" else 1
                    t = make_vlm_query(cfg, subquery, action_class)

                    if cfg.stage4.vlm_type == "coagent":
                        raise NotImplementedError
                    
                    elif cfg.stage4.vlm_type == "llava-ov":
                        with torch.no_grad():
                            conversation = [{
                                "role": "user",
                                "content": [{"type": "image", "url": frame}, {"type": "text", "text": t}],
                            }]
                            inputs = model_dict["processor"].apply_chat_template(
                                conversation, 
                                add_generation_prompt=True, 
                                tokenize=True, 
                                return_dict=True, 
                                return_tensors="pt"
                            ).to(f"cuda", torch.float16)

                            yes_idx = model_dict["processor"].tokenizer.convert_tokens_to_ids("yes")
                            no_idx = model_dict["processor"].tokenizer.convert_tokens_to_ids("no")

                            outputs = model_dict["model"].generate(
                                **inputs, 
                                max_new_tokens=model_dict["max_length"],
                                return_dict_in_generate=True,
                                output_scores=True
                            )
                            
                            logits = outputs.scores[0]
                            yes_no_logits = logits[0, [yes_idx, no_idx]]
                            yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)
                            p = yes_no_probs[0].item()

                    elif cfg.stage4.vlm_type == "qwen2-vl":
                        with torch.no_grad():
                            conversation = [{
                                "role": "user",
                                "content": [{"type": "image", "image": frame}, {"type": "text", "text": t}],
                            }]
                            text_prompt = model_dict["processor"].apply_chat_template(
                                conversation, 
                                add_generation_prompt=True,
                                # tokenize=True,
                            )
                            inputs = model_dict["processor"](
                                text=[text_prompt], 
                                images=[frame], 
                                padding=True, 
                                return_tensors="pt"
                            ).to("cuda", torch.float16)

                            yes_idx = 9454
                            no_idx = 2753
                            
                            outputs = model_dict["model"].generate(
                                **inputs, 
                                max_new_tokens=model_dict["max_length"],
                                return_dict_in_generate=True,
                                output_scores=True
                            )

                            logits = outputs.scores[0]
                            yes_no_logits = logits[0, [yes_idx, no_idx]]
                            yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)
                            p = yes_no_probs[0].item()

                    elif cfg.stage4.vlm_type == "pixtral":
                        with torch.no_grad():
                            conversation = [{
                                "role": "user",
                                "content": [{"type": "image", "url": frame}, {"type": "text", "content": t}],
                            }]
                            inputs = model_dict["processor"].apply_chat_template(
                                conversation, 
                                add_generation_prompt=True, 
                                tokenize=True, 
                                return_dict=True, 
                                return_tensors="pt"
                            ).to(f"cuda", torch.float16)

                            yes_idx = model_dict["processor"].tokenizer.convert_tokens_to_ids("yes")
                            no_idx = model_dict["processor"].tokenizer.convert_tokens_to_ids("no")
                            
                            outputs = model_dict["model"].generate(
                                **inputs, 
                                max_new_tokens=model_dict["max_length"],
                                return_dict_in_generate=True,
                                output_scores=True
                            )

                            logits = outputs.scores[0]
                            yes_no_logits = logits[0, [yes_idx, no_idx]]
                            yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)
                            p = yes_no_probs[0].item()
                    
                    else:
                        raise NotImplementedError

                    scores[i, label_map[action_class], k] = p

        _save_scores(cfg, scores, video_name.split(".")[0], output_dir)


def _process_single_video_wvlm_v2(
    cfg : DictConfig,
    video_name : str, 
    model_dict : Dict[str, Any],
    output_dir : str, 
    video_reader : Any,
    query_dict : Dict[str, Dict[str, Any]],
    label_map : Dict[str, int],
):
    pass