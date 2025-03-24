"""
Check data / features / scores are saved correctly.
"""

import glob
import hydra
import json
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import dotenv_values
from omegaconf import DictConfig

from helpers.al_dataset import read_data
from stage3_filter_action_classes import _path 


@hydra.main(
    version_base="1.1", config_path="conf", config_name="config.yaml"
)
def main(cfg : DictConfig):
    env = dotenv_values()
        
    OUTPUT_DIR = env.get("OUTPUT_DIR", None) 
    LLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "llm_outputs", cfg.stage1.llm_type, cfg.dataset.name)
    CLIP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "clip_outputs", cfg.dataset.name, cfg.split)
    FILT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "filter_outputs", cfg.dataset.name, cfg.split)
    VLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vlm_outputs", cfg.stage4.vlm_type, cfg.dataset.name, cfg.split)
    VIDEO_DIR = cfg.dataset.video_root
    ANN_DIR = cfg.dataset.ann_root

    if cfg.dataset.name == "thumos14":
        VIDEO_DIR = os.path.join(VIDEO_DIR, cfg.split)
        ANN_DIR = os.path.join(ANN_DIR, f"annotations_{cfg.split}")

    video_names, gt_df, label_map = read_data(
        cfg.dataset.name, ann_dir=ANN_DIR, split=cfg.split, max_videos=cfg.max_videos
    )
    if cfg.example_id != '':
        video_names = [v for v in video_names if str(cfg.example_id) in v]

    logging.info(f"For {cfg.dataset.name} dataset: {len(video_names)} videos found.")

    logging.info("Stage 1:")
    message_exists = os.path.exists(os.path.join(LLM_OUTPUT_DIR, f"message_{cfg.score_strategy}.json"))
    logging.info(f"Does message file (from {cfg.stage1.llm_type}) exist? {message_exists}")
    # check if all classes are present in the message file
    if message_exists:
        with open(os.path.join(LLM_OUTPUT_DIR, f"message_{cfg.score_strategy}.json"), "r") as f:
            vlm_queries = json.load(f)
        all_classes = set(label_map.keys())
        missing_classes = all_classes - set(vlm_queries["content"].keys())
        _missing_classes = len(missing_classes) > 0
        logging.info(f"Are all classes there? {not _missing_classes}")
        if _missing_classes:
            logging.info(f"Missing classes: {missing_classes}")

    logging.info("Stage 2:")
    img_feat_paths = [
        os.path.basename(path) 
        # for path in glob.glob(os.path.join(CLIP_OUTPUT_DIR, "img_feats", cfg.stage2.clip_model.arch, f"*_{cfg.stage4.fps}.pt"))
        for path in glob.glob(os.path.join(CLIP_OUTPUT_DIR, "img_feats", cfg.stage2.clip_model.arch, f"*.pt"))
    ]
    txt_feat_paths = [
        os.path.basename(path) 
        # for path in glob.glob(os.path.join(CLIP_OUTPUT_DIR, "txt_feats", cfg.stage2.clip_model.arch, f"*_{cfg.stage4.fps}.pt"))
        for path in glob.glob(os.path.join(CLIP_OUTPUT_DIR, "txt_feats", cfg.stage2.clip_model.arch, f"*.pt"))
    ]
    complete = len(img_feat_paths) == len(video_names) and len(txt_feat_paths) == len(video_names)
    logging.info(f"Have all CLIP features been extracted (using {cfg.stage2.clip_model.arch})? {complete}")
    if not complete: 
        # missing_videos = [v for v in video_names if f"{v}.pt" not in img_feat_paths]
        logging.info(f"Number of features: {len(img_feat_paths)} / {len(video_names)}")
        # logging.info(f"Missing videos: {missing_videos}")

    logging.info("Stage 3:")
    num_ac_1hot = len(glob.glob(os.path.join(FILT_OUTPUT_DIR, f"*_fac_{cfg.stage2.clip_model.arch[0]}{cfg.stage3.n_frames}{str(cfg.stage3.use_gt)[0]}.npy")))
    complete = num_ac_1hot == len(video_names)
    if cfg.stage3.use_gt:
        logging.info(f"Have all action classes been filtered (using GT)? {complete}")
    else:
        logging.info(f"Have all action classes been filtered? {complete}")
    if not complete:
        # missing_videos = [v for v in video_names if f"{v}_ac_1hot.npy" not in os.listdir(FILT_OUTPUT_DIR)]
        logging.info(f"Number of filtered results: {num_ac_1hot} / {len(video_names)}")
        # logging.info(f"Missing videos: {missing_videos}")

    logging.info("Stage 4:")
    num_outputs = len(glob.glob(os.path.join(VLM_OUTPUT_DIR, f"*_vlm_scores_{cfg.score_strategy}{cfg.stage4.fps}_{cfg.stage1.llm_type}{_path(cfg)}")))
    # num_outputs = len(glob.glob(os.path.join(VLM_OUTPUT_DIR, f"*_vlm_scores_{cfg.score_strategy}_{cfg.stage1.llm_type}{_path(cfg)}")))
    complete = num_outputs == len(video_names)
    logging.info(f"Have all VLM soft-scores been generated? {complete}")
    if not complete:
        # missing_videos = [v for v in video_names if f"{v}.npy" not in os.listdir(VLM_OUTPUT_DIR)]
        logging.info(f"Number of VLM outputs: {num_outputs} / {len(video_names)}")
        # logging.info(f"Missing videos: {missing_videos}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()