"""
Helper functions for visualization.
"""
from typing import Dict

import logging
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.gridspec as gridspec

from decord import cpu, VideoReader
from omegaconf import DictConfig
from stage4_fine_grain_zsal import _load_scores
from stage5_gen_proposals import _get_action_boundaries, _modified_minmax_norm


def stage2_figure():
    pass


def stage3_figure():
    pass


def stage45_figure(
    cfg : DictConfig,
    video_name : str,
    video_dir : str,
    figure_dir : str,
    score_dir : str,
    gt_df : pd.DataFrame,
    label_map : Dict[str, int],
    pred_df : pd.DataFrame = None,
):
    """
    Show start and end scores for a given video.
    """
    colors = {v: mcolors.to_hex(np.random.rand(3,)) for v in label_map.values()}
    fontsize = 20

    if pred_df is not None:
        pred_df = pred_df[pred_df['video-id'] == video_name]
    gt_df = gt_df[gt_df['video-id'] == video_name]

    if pred_df is not None and pred_df.empty: 
        logging.warning(f"No predictions for video {video_name}.")
        pred_df = None

    if gt_df.empty:
        logging.warning(f"No ground truth data for video {video_name}.")
        return

    unique_labels = sorted(set(gt_df['label']).union(set(pred_df['label'] if pred_df is not None else [])))
    label_y_positions = {label: i for i, label in enumerate(unique_labels)}

    scores = _load_scores(cfg, video_name, score_dir)
    non_zero_columns = np.any(scores != 0, axis=0)
    non_zero_columns = np.all(non_zero_columns, axis=-1)
    
    scores = scores[:, non_zero_columns, :]
    scores = _modified_minmax_norm(scores, cfg.stage5.mm_eps)
    non_zero_indices = np.where(non_zero_columns)[0]
    
    N = scores.shape[1] * 2
    total_rows = N + 2 if pred_df is not None else N + 1
    M = 121

    # Create figure with custom GridSpec
    fig = plt.figure(figsize=(12, 6))
    
    # Define height ratios - make the last subplot(s) taller
    height_ratios = [1] * (total_rows - 1)  # Regular height for score plots
    height_ratios += [5] * 1  # Triple height for predictions
    
    gs = gridspec.GridSpec(total_rows, 1, height_ratios=height_ratios)
    axs = [fig.add_subplot(gs[i]) for i in range(total_rows)]

    for k in range(scores.shape[1]):
        starts, ends = _get_action_boundaries(scores[:, k, :], p=cfg.stage5.top_p)
 
        ax1 = axs[2*k]
        ax2 = axs[2*k + 1]
        tick_positions = range(0, M, 8)

        ax1.bar(range(M), scores[:M, k, 0], color='dodgerblue', alpha=0.8)
        ax1.set_ylim(0, 1.1) 
        ax1.set_yticks([])
        ax1.set_xlim(0, M)
        ax1.set_xticks([])
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylabel('start\nquery', fontsize=fontsize - 10)

        ax2.bar(range(M), scores[:M, k, 1], color='darkorange', alpha=0.8)
        ax2.set_ylim(0, 1.1) 
        ax2.set_yticks([])
        ax2.set_xlim(0, M) 
        ax2.set_xticks([])
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylabel('end\nquery', fontsize=fontsize - 10)

        for start in starts:
            if start < M:
                ax1.axvline(x=start+0.01, color='black', alpha=0.5)
                ax1.plot(start+0.01, 1.05, marker='1', markersize=9, color='red')
        for end in ends:
            if end < M:
                ax2.axvline(x=end+0.01, color='black', alpha=0.5)
                ax2.plot(end+0.01, 1.05, marker='1', markersize=9, color='red')

    ax3 = axs[-2] if pred_df is not None else axs[-1]

    for _, row in gt_df.iterrows():
        if row['t-end'] > M:
            continue
        ax3.barh(
            label_y_positions[row['label']], width=row['t-end'] - row['t-start'], left=row['t-start'],
            color=colors.get(row['label'], 'gray'), edgecolor='black', 
        )

        gt_text = list(label_map.keys())[row['label']]
        gt_text = ''.join([i for i in gt_text if i.isupper()])

        ax3.text(
            x=row['t-start'] + (row['t-end'] - row['t-start']) / 2, 
            y=label_y_positions[row['label']], 
            s=gt_text, 
            va='center',  
            ha='center',
            fontsize=6,  
        )

    ax3.set_yticks([])
    ax3.set_xlim(0, M) 
    ax3.set_xticks([])
    ax3.set_ylabel(r'$a$', fontsize=fontsize)
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = axs[-1] if pred_df is not None else None
    if ax4 is not None:
        for i, row in pred_df.iterrows():
            ax4.barh(
                i, height=1, width=row['t-end'] - row['t-start'], left=row['t-start'],
                color=colors.get(row['label'], 'gray'), alpha=row['score'],
                edgecolor='black' if row['score'] > 0.5 else 'none',
            )
            if row['t-start'] > M:
                break

        ax4.set_yticks([])
        ax4.set_xlim(0, M) 
        ax4.set_xticks(tick_positions) 
        ax4.tick_params(axis='x', labelsize=fontsize - 10)
        ax4.set_ylabel(r'$\hat{a}$', fontsize=fontsize)
        ax4.set_xlabel('Time (s)', fontsize=fontsize - 5)
        ax4.spines['left'].set_visible(False)
        ax4.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    # plt.tight_layout()
    plt.show()

    if pred_df is None:
        os.makedirs(f"{figure_dir}/stage4", exist_ok=True)
        plt.savefig(f"{figure_dir}/stage4/{video_name}.png", dpi=300)
    else:
        os.makedirs(f"{figure_dir}/stage5", exist_ok=True)
        plt.savefig(f"{figure_dir}/stage5/{video_name}.png", dpi=300)


def precrec_figure(
    figure_dir : str,
):
    n_frames = [1, 4, 8, 16, 32]
    precs_at5 = [0.143, 0.220, 0.226, 0.225, 0.225]
    recs_at5 = [0.615, 0.943, 0.972, 0.964, 0.968]
    precs_at3 = [0.193, 0.311, 0.332, 0.330, 0.333]
    recs_at3 = [0.498, 0.802, 0.854, 0.850, 0.858]

    plt.figure(figsize=(6, 5))
    plt.plot(n_frames, precs_at5, marker='o', linestyle='-', color='lightgrey')
    plt.plot(n_frames, recs_at5, marker='d', linestyle='-', color='lightgrey')
    plt.plot(n_frames, precs_at3, marker='o', linestyle='-', color='black')
    plt.plot(n_frames, recs_at3, marker='d', linestyle='-', color='black')

    plt.xticks(n_frames, fontsize=16)
    plt.ylim(0, 1.0)
    plt.xlabel('Frames sampled', fontsize=16)
    plt.grid(True, alpha=0.5)
    plt.legend(['Prec@5', 'Rec@5', 'Prec@3', 'Rec@3'], fontsize=12, ncol=2,loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/precrec_fig.png", dpi=300)


def lambda_figure(
    figure_dir : str,
):
    lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    th14_mAP_scores = [11.52, 13.74, 15.73, 15.89, 16.00, 15.68, 15.54, 15.22, 15.08, 14.69, 14.62]

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, th14_mAP_scores, marker='o', linestyle='-', color='dodgerblue')
    plt.ylim(min(th14_mAP_scores) - 1.0, max(th14_mAP_scores) + 1.0)
    plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel('Average mAP', fontsize=16)
    plt.grid(True, alpha=0.5)
    plt.savefig(f"{figure_dir}/lambda_fig.png", dpi=300)


if __name__ == "__main__":
    # lambda_figure('/pasteur/u/josaklil/code/ZEAL/outputs/figures')
    precrec_figure('/pasteur/u/josaklil/code/ZEAL/outputs/figures')