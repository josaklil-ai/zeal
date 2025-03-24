"""
Stage 5: Generate proposals from VLM soft scores.
"""

import logging
import numpy as np
import pandas as pd
import os

from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Dict

from helpers.eval import segment_iou
from helpers.softnms import softnms_1d
from stage2_coarse_grain_zsal import _load_clip_features
from stage3_filter_action_classes import _path


def _modified_minmax_norm(
    scores : np.ndarray,
    eps : float = 1e-6,
):
    norm_scores = (scores - scores.min(axis=0)) / (scores.max(axis=0) - scores.min(axis=0))
    scaled_norm_scores = norm_scores * (1 - 2 * eps) + eps

    return scaled_norm_scores


def _get_action_boundaries(
    scores : np.ndarray, 
    p : float = 0.05,
    eps: int = 5,
):
    start_scores = scores[:, 0]
    end_scores = scores[:, 1]

    if np.isnan(start_scores).any() or np.isnan(end_scores).any():
        return np.array([0]), np.array([0])

    if int(p * len(start_scores)) < 1:
        return np.array([0]), np.array([len(start_scores) - 1])

    min_start_score = np.sort(start_scores)[::-1][:int(p * len(start_scores))][-1]
    min_end_score = np.sort(end_scores)[::-1][:int(p * len(end_scores))][-1]

    start_candidates = np.where(start_scores >= min_start_score)[0]
    end_candidates = np.where(end_scores >= min_end_score)[0]

    # gradually increase p until we get at least one start and one end
    while len(start_candidates) == 0 or len(end_candidates) == 0:
        p += 0.01
        min_start_score = np.sort(start_scores)[::-1][:int(p * len(start_scores))][-1]
        min_end_score = np.sort(end_scores)[::-1][:int(p * len(end_scores))][-1]

        start_candidates = np.where(start_scores >= min_start_score)[0]
        end_candidates = np.where(end_scores >= min_end_score)[0]

    # Apply NMS to get diverse start and end points
    starts = []
    start_scores_candidates = start_scores[start_candidates]
    sorted_start_candidates = start_candidates[np.argsort(start_scores_candidates)[::-1]]
    
    for candidate in sorted_start_candidates:
        if all(abs(candidate - s) >= eps for s in starts) or not starts:
            starts.append(candidate)
    
    ends = []
    end_scores_candidates = end_scores[end_candidates]
    sorted_end_candidates = end_candidates[np.argsort(end_scores_candidates)[::-1]]
    
    for candidate in sorted_end_candidates:
        if all(abs(candidate - e) >= eps for e in ends) or not ends:
            ends.append(candidate)

    return np.sort(starts), np.sort(ends)


def _get_interval_scores(
    cfg : DictConfig,
    video_id : str,
    clip_output_dir : str,
    scores : np.ndarray, 
    intervals : np.ndarray,
    label : int,
):
    img_feats = _load_clip_features(cfg, video_id, clip_output_dir, "img_feats")
    txt_feats = _load_clip_features(cfg, video_id, clip_output_dir, "txt_feats")

    interval_txt_feat = txt_feats[label][np.newaxis, :]
    
    interval_scores = []
    for interval in intervals:
        start, end = interval
        start_score, end_score = scores[start, 0], scores[end, 1]

        interval_img_feats = img_feats[start:end+1]
        sims = np.array([
            np.dot(interval_img_feats[i], interval_txt_feat[0]) 
            for i in range(len(interval_img_feats))
        ])
        
        loc_conf_score = start_score + end_score
        avg_clip_conf_score = np.mean(sims)

        s = cfg.stage5._lambda * loc_conf_score + (1 - cfg.stage5._lambda) * avg_clip_conf_score
        interval_scores.append(s)
    
    return np.array(interval_scores)


def _get_valid_intervals(
    cfg : DictConfig,
    starts : np.ndarray,
    ends : np.ndarray,
    N : int,
):
    max_length = cfg.stage5.max_interval_len * N

    valid_intervals = []
    for i in range(len(starts)):
        stop = False
        for j in range(len(ends)):
            if starts[i] < ends[j] and ends[j] - starts[i] <= max_length and not stop:
                valid_intervals.append((starts[i], ends[j]))
            if i < len(starts) - 1:
                if ends[j] > starts[i+1] and not stop:
                    stop = True

    if len(valid_intervals) == 0:
        return np.array([(starts[0], ends[0])])

    return np.array(valid_intervals)


def _merge_intervals(
    intervals: np.ndarray,
    s_k: np.ndarray,
    merge_thresh: float = 0.9,
    score_thresh: float = 0.75,
):
    idxs = np.argsort(intervals[:, 0])
    intervals = intervals[idxs]
    s_k = s_k[idxs]

    merged_intervals = []
    merged_scores = []

    for i, interval in enumerate(intervals):
        if i == 0:
            merged_intervals.append(interval)
            merged_scores.append(s_k[i])
            continue

        prev_interval = merged_intervals[-1]
        prev_score = merged_scores[-1]

        if segment_iou(prev_interval, interval[None, :]) > merge_thresh:
            # Take the earlier start time and later end time when merging
            start_time = min(prev_interval[0], interval[0])
            end_time = max(prev_interval[1], interval[1])
            merged_intervals[-1] = np.array([start_time, end_time])
            merged_scores[-1] = (prev_score + s_k[i]) / 2
        else:
            merged_intervals.append(interval)
            merged_scores.append(s_k[i])

    filtered = [
        (interval, score) 
        for interval, score in zip(merged_intervals, merged_scores) 
        if score >= score_thresh
    ]
    merged_intervals, merged_scores = zip(*filtered)

    return np.array(merged_intervals), np.array(merged_scores)


def _get_action_intervals(
    cfg : DictConfig, 
    video_id : str,
    clip_output_dir : str,
    scores : np.ndarray,
    label : int,
):
    # 1) Normalize scores to [0, 1]
    scores = _modified_minmax_norm(scores, cfg.stage5.mm_eps)

    # 2) Get start and end candidates
    starts, ends = _get_action_boundaries(scores, p=cfg.stage5.top_p)
    assert len(starts) > 0 and len(ends) > 0, "No start or end candidates found."
    
    # 3) Get all valid intervals and initial interval scores
    action_intervals = _get_valid_intervals(cfg, starts, ends, len(scores))
    if len(action_intervals) == 0:
        logging.warning("No valid intervals found with candidates.")
    s_k = _get_interval_scores(
        cfg, video_id, clip_output_dir, scores, action_intervals, label
    )

    # 4) Apply NMS to get final intervals
    pre_nms = len(action_intervals)
    action_intervals, s_k = softnms_1d(action_intervals, s_k, 
        Nt=cfg.stage5.nms_thresh, 
        sigma=cfg.stage5.nms_sigma, 
        method=cfg.stage5.nms_method, 
    )
    post_nms = len(action_intervals)
    # logging.info(f"NMS reduced {pre_nms} intervals to {post_nms} intervals.")
  
    # 5) Merge intervals if too much overlap 
    if len(action_intervals) > 1:
        pre_merge = len(action_intervals)
        action_intervals, s_k = _merge_intervals(
            action_intervals, s_k, cfg.stage5.merge_thresh, cfg.stage5.score_thresh
        )
        post_merge = len(action_intervals)
        # logging.info(f"Merged {pre_merge} intervals into {post_merge} intervals.")
    
    return action_intervals, s_k


def _process_score_path(
    cfg : DictConfig,
    score_path : Path,
    clip_output_dir : str,
):
    video_id = score_path.stem.split("_vlm")[0]
    scores = np.load(score_path)

    rows = []

    if scores.shape[0] <= 2:
        logging.warning(f"Video {video_id} has few frames is likely corrupted.")
        return rows

    for k in range(scores.shape[1]):
        if scores[:, k, :].sum() == 0:
            continue
        if not cfg.stage3.use_gt:
            if scores[:, k, :].mean() < 0.1:
                continue

        intervals, s_k = _get_action_intervals(cfg, video_id, clip_output_dir, scores[:, k, :], k)
        for interval, s in zip(intervals, s_k):
            row = {
                "video-id": video_id, 
                "t-start": interval[0], "t-end": interval[1], 
                "label": k, 
                "score": s,
            }
            rows.append(row)  

    return rows


def gen_proposals(
    cfg : DictConfig,
    clip_output_dir : str,
    output_dir : str,
    label_map : Dict[str, int],
):
    ext = f"_{cfg.score_strategy}{cfg.stage4.fps}_{cfg.stage1.llm_type}{_path(cfg)}"
    # ext = ext.replace("_E", "_V")     # TODO: if EVA-CLIP used for actionness, adjust extension to get scores
    logging.info(f"Loading scores with this config: {ext}")

    score_paths = [path for path in Path(output_dir).rglob(f"*_vlm_scores{ext}")]
    # score_paths = np.random.choice(score_paths, 500, replace=False)

    if cfg.stage5.show_id != '' and not cfg.evaluate:
        score_paths = [path for path in score_paths if str(cfg.stage5.show_id) in path.stem]

    logging.info(f"Processing {len(score_paths)} videos from {cfg.dataset.name}...")

    all_rows = []
    pred_count = 0
    for score_path in tqdm(score_paths):
        rows = _process_score_path(cfg, score_path, clip_output_dir)
        if len(rows) == 0:
            logging.warning(f"No proposals found for {score_path.stem.split('_vlm')[0]}.")
        all_rows.extend(rows)
        pred_count += len(rows)

    logging.info(f'Avg. pred count per video: {pred_count/len(score_paths):.2f}')

    pred_df = pd.DataFrame(all_rows, columns=["video-id", "t-start", "t-end", "label", "score"])
    
    return pred_df