"""
Stage 3: Filter Action Class Space 

To reduce computational overhead from the VLM agent, we filter the set of possible queries by
prompting the agent with the list of action classes and selecting only the top-k most likely.
"""
import logging
import numpy as np
import os
import pandas as pd
import textwrap 

from omegaconf import DictConfig
from typing import Any, Dict, List

from stage2_coarse_grain_zsal import _load_clip_features


def _create_filter_vlm_query(
    action_classes : List[str],
):
    query = textwrap.dedent(f"""
        Which of the following action classes is this frame most likely depicting? {action_classes} 
        Give at least 1 but no more than 2 action classes.
    """)
    return query


def _rank_action_classes(
    cfg : DictConfig,
    video_name : str,
    clip_output_dir : str,
    query_dict : Dict[str, Dict[str, Any]],
    n_frames : int = 16,
):
    img_features = _load_clip_features(cfg, video_name.split(".")[0], clip_output_dir, "img_feats")
    txt_features = _load_clip_features(cfg, video_name.split(".")[0], clip_output_dir, "txt_feats")

    sampled_indices = np.linspace(0, len(img_features) - 1, n_frames).astype(int)
    sampled_img_features = img_features[sampled_indices].numpy()
    avg_img_features = np.mean(sampled_img_features, axis=0)

    sims = np.dot(avg_img_features, txt_features.T)

    # mean_sim = np.mean(sims)
    # std_sim = np.std(sims)
    # threshold = mean_sim + 1 * std_sim

    # selected_indices = np.where(sims >= threshold)[0]

    # if len(selected_indices) == 0:
    #     selected_indices = [np.argmax(sims)]
    # elif len(selected_indices) > cfg.dataset.max_classes:
    #     top_indices = np.argsort(sims[selected_indices])[::-1][:cfg.dataset.max_classes]
    #     selected_indices = selected_indices[top_indices]

    selected_indices = np.argsort(sims)[::-1][:cfg.dataset.max_classes]
    
    keys = sorted(query_dict.keys())
    action_classes = [keys[i] for i in selected_indices]

    return action_classes


def _filter_action_class(
    cfg : DictConfig,
    video_name : str,
    clip_output_dir : str,
    output_dir : str,
    video_reader : Any,
    query_dict : Dict[str, Dict[str, Any]],
    label_map : Dict[str, int],
    gt_df : pd.DataFrame,
):
    ac_1hot = _load_ac_1hot(cfg, video_name, output_dir)
    
    if ac_1hot is not None and not cfg.stage3.overwrite:
        filtered_query_dict = {
            k: v for k, v in query_dict.items() 
            if ac_1hot[label_map[k]] == 1
        }
        logging.info(f"Preloaded filtered action class dict for {video_name}.")
    elif cfg.stage3.use_gt:
        filtered_query_dict = {
            k: v for k, v in query_dict.items() 
            if label_map[k] in gt_df[gt_df["video-id"] == video_name]['label'].values
        }
        ac_1hot = np.zeros(len(label_map))
        for k in filtered_query_dict:
            ac_1hot[label_map[k]] = 1

        _save_ac_1hot(cfg, ac_1hot, video_name, output_dir)
        logging.info(f"Filtered action class dict for {video_name} using GT.")
    else:
        action_classes = _rank_action_classes(cfg, video_name, clip_output_dir, query_dict, cfg.stage3.n_frames)
        
        filtered_query_dict = {
            k: v for k, v in query_dict.items() 
            if k in action_classes
        }
        ac_1hot = np.zeros(len(label_map))
        for k in filtered_query_dict:
            ac_1hot[label_map[k]] = 1

        _save_ac_1hot(cfg, ac_1hot, video_name, output_dir)
        logging.info(f"Filtered action class dict for {video_name}.")

    return filtered_query_dict


def _path(
    cfg : DictConfig,
    video_name : str = None,
):
    suffix = f"_fac_{cfg.stage2.clip_model.arch[0]}{cfg.stage3.n_frames}{str(cfg.stage3.use_gt)[0]}.npy"
    return f"{video_name}{suffix}" if video_name is not None else suffix


def _save_ac_1hot(
    cfg : DictConfig, 
    ac_1hot : np.ndarray,
    video_name : str, 
    output_dir : str,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, _path(cfg, video_name)), ac_1hot)


def _load_ac_1hot(
    cfg : DictConfig,
    video_name : str,
    output_dir : str,
):
    path = os.path.join(output_dir, _path(cfg, video_name))

    if os.path.exists(path):
        return np.load(path)
    else:
        return None


def compute_filter_prec_rec(
    cfg : DictConfig,
    output_dir : str,
    gt_df : pd.DataFrame,
):
    tp, fp, fn = 0, 0, 0

    for file in os.listdir(output_dir):
        if file.endswith(_path(cfg)):
            video_id = file.split("_fac_")[0]
            ac_1hot = np.load(os.path.join(output_dir, file))

            pred_ac = np.where(ac_1hot == 1)[0]
            gt_ac = np.unique(gt_df[gt_df["video-id"] == video_id]['label'].values)
            
            tp += len(np.intersect1d(pred_ac, gt_ac))
            fp += len(np.setdiff1d(pred_ac, gt_ac))
            fn += len(np.setdiff1d(gt_ac, pred_ac))

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    return prec, rec