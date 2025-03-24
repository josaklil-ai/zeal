"""
Run ZSAL framework and evaluate.
"""
import torch
import decord, glob, hydra, json, logging, numpy, open_clip, os, pandas, time, transformers, warnings
warnings.filterwarnings("ignore")

from decord import cpu
from dotenv import dotenv_values
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration, 
    Qwen2VLForConditionalGeneration,
)
transformers.logging.set_verbosity_error()
from typing import Any, Dict

from helpers.al_dataset import read_data
from helpers.eval import evaluate
from helpers.figures import stage45_figure

from stage1_gen_vlm_queries import gen_vlm_queries
from stage2_coarse_grain_zsal import _process_single_video_wclip
from stage3_filter_action_classes import _filter_action_class, compute_filter_prec_rec, _path
from stage4_fine_grain_zsal import _process_single_video_wvlm, _process_single_video_wvlm_v2
from stage5_gen_proposals import gen_proposals


MODELS = {}

def initialize_model_vlm(
    gpu_idx : int,
    model_type : str,
):
    torch.cuda.set_device(gpu_idx)

    if model_type == "cogagent":
        raise NotImplementedError

    elif model_type == "llava-ov":
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
            use_fast=True
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=f"cuda:{gpu_idx}"
        )
        model.eval()
        model_dict = {
            "model": model,
            "processor": processor,
            "max_length": 128
        }

    elif model_type == "qwen2-vl":
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            use_fast=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map=f"cuda:{gpu_idx}"
        )
        model.eval()
        model_dict = {
            "model": model,
            "processor": processor,
            "max_length": 128
        }

    elif model_type == "pixtral":
        processor = AutoProcessor.from_pretrained(
            "mistral-community/pixtral-12b", 
            use_fast=True
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            "mistral-community/pixtral-12b", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map=f"cuda:{gpu_idx}"
        )
        model.eval()
        model_dict = {
            "model": model,
            "processor": processor,
            "max_length": 128
        }

    else:
        raise ValueError(f"Model type {model_type} not supported.")

    return model_dict


def initialize_model_clip(
    cfg: DictConfig,
    gpu_idx : int,
):
    torch.cuda.set_device(gpu_idx)

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.stage2.clip_model.arch, pretrained=cfg.stage2.clip_model.weights
    )
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer(cfg.stage2.clip_model.arch)
    logging.info(f"Model initialized on GPU {gpu_idx}.")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "preprocess": preprocess,
    }


def process_video(
    cfg : DictConfig, 
    video_name : str, 
    gpu_idx : int,
    clip_output_dir : str,
    filter_output_dir : str,
    vlm_output_dir : str, 
    video_dir : str, 
    query_dict : Dict[str, Dict[str, Any]],
    label_map : Dict[str, int],
    gt_df : pandas.DataFrame,
):
    torch.cuda.set_device(gpu_idx)
    
    video_path = None
    for ext in [".mp4", ".mkv", ".webm"]:
        if video_path is None:
            p = glob.glob(os.path.join(video_dir, f"{video_name}{ext}"), recursive=False)
            video_path = p[0] if len(p) > 0 else None
    assert video_path is not None, logging.ERROR(f"Video {video_name} not found.")

    video_reader = decord.VideoReader(str(video_path), ctx=cpu(0), num_threads=4)

    if cfg.stage2.enable:
        _process_single_video_wclip(
            cfg, video_name, MODELS[gpu_idx], clip_output_dir, video_reader, query_dict
        )
        return

    if cfg.stage3.enable:
        query_dict = _filter_action_class(
            cfg, video_name, clip_output_dir, filter_output_dir, video_reader, query_dict, label_map, gt_df
        )

    if cfg.stage4.enable:
        if cfg.score_strategy == "SEQ":
            _process_single_video_wvlm(
                cfg, video_name, MODELS[gpu_idx], vlm_output_dir, video_reader, query_dict, label_map
            )
        elif cfg.score_strategy == "PHR":
            _process_single_video_wvlm_v2(
                cfg, video_name, MODELS[gpu_idx], vlm_output_dir, video_reader, query_dict, label_map
            )
        elif cfg.score_strategy == "PHR-SEQ":
            raise NotImplementedError


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def main(cfg : DictConfig):
    env = dotenv_values()

    time_dict = {}
    start = time.time()
    N_GPUS = torch.cuda.device_count()
    
    # Create output directories
    OUTPUT_DIR = env.get("OUTPUT_DIR", None)
    LOG_DIR = env.get("LOG_DIR", None)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    LLM_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR, "llm_outputs", cfg.stage1.llm_type, cfg.dataset.name
    )
    CLIP_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR, "clip_outputs", cfg.dataset.name, cfg.split
    )
    FILT_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR, "filter_outputs", cfg.dataset.name, cfg.split
    ) 
    VLM_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR, "vlm_outputs", cfg.stage4.vlm_type, cfg.dataset.name, cfg.split
    )
    os.makedirs(LLM_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FILT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VLM_OUTPUT_DIR, exist_ok=True)

    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    VIDEO_DIR = cfg.dataset.video_root
    ANN_DIR = cfg.dataset.ann_root

    if cfg.dataset.name == "thumos14":
        VIDEO_DIR = os.path.join(VIDEO_DIR, cfg.split)
        ANN_DIR = os.path.join(ANN_DIR, f"annotations_{cfg.split}")

    assert os.path.isdir(VIDEO_DIR), f"Video directory {VIDEO_DIR} does not exist."      
    assert os.path.isdir(ANN_DIR), f"Annotation directory {ANN_DIR} does not exist."

    # Read AL dataset
    video_names, gt_df, label_map = read_data(
        cfg.dataset.name, ann_dir=ANN_DIR, split=cfg.split, max_videos=cfg.max_videos
    )
    if cfg.example_id != '':
        video_names = [v for v in video_names if str(cfg.example_id) in v]

    if cfg.mp.block_id >= 0:
        block = len(video_names) // cfg.mp.num_blocks
        video_names = sorted(video_names)
        video_names = video_names[cfg.mp.block_id * block : (cfg.mp.block_id + 1) * block]

    time_dict["read_data"] = time.time() - start
    start = time.time()

    # Generate VLM queries
    vlm_query_path = os.path.join(LLM_OUTPUT_DIR, f"message_{cfg.score_strategy}.json")
    if cfg.stage1.enable or not os.path.exists(vlm_query_path):
        gen_vlm_queries(cfg, LLM_OUTPUT_DIR)
        logging.info("Generated VLM queries.")
    else:
        with open(vlm_query_path, "r") as f:
            vlm_queries = json.load(f)
        logging.info("Preloaded VLM queries.")

    time_dict["gen_vlm_queries"] = time.time() - start
    start = time.time()

    # Initialize CLIP and VLM models
    if cfg.stage2.enable:
        for gpu_idx in range(N_GPUS):
            MODELS[gpu_idx] = initialize_model_clip(cfg, gpu_idx)

        time_dict["init_clips"] = time.time() - start
        start = time.time()
    
    elif cfg.stage4.enable:
        for gpu_idx in range(N_GPUS):
            MODELS[gpu_idx] = initialize_model_vlm(gpu_idx, cfg.stage4.vlm_type)
        
        time_dict["init_vlms"] = time.time() - start
        start = time.time()

    # Process videos
    if cfg.stage2.enable or cfg.stage3.enable or cfg.stage4.enable:
        logging.info(f"Processing {len(video_names)} videos.")

        num_workers = N_GPUS if N_GPUS > 0 else os.cpu_count()
        for idx, video_name in tqdm(enumerate(video_names), total=len(video_names)):
            process_video(
                cfg, video_name, idx % num_workers, 
                CLIP_OUTPUT_DIR, FILT_OUTPUT_DIR, VLM_OUTPUT_DIR, VIDEO_DIR, 
                vlm_queries["content"], label_map, gt_df
            )
            logging.debug(f"Succesfully processed video {video_name}.")

    time_dict["process_videos"] = time.time() - start
    start = time.time()

    # Evaluation
    if cfg.stage3.eval:
        ac_prec, ac_rec = compute_filter_prec_rec(cfg, FILT_OUTPUT_DIR, gt_df)
        logging.info(f"Filtering precision: {ac_prec:3f}, recall: {ac_rec:3f}.")
        
        time_dict["stage3_eval"] = time.time() - start
        start = time.time()

    if cfg.stage4.show_results:
        temp_v = [v for v in video_names if str(cfg.stage4.show_id) in v][0]
        stage45_figure(cfg, temp_v, VIDEO_DIR, FIGURE_DIR, VLM_OUTPUT_DIR, gt_df, label_map)

    if cfg.evaluate or cfg.stage5.enable or cfg.stage5.show_results:  
        pred_df = gen_proposals(cfg, CLIP_OUTPUT_DIR, VLM_OUTPUT_DIR, label_map)
        logging.info("Generated proposals.")

        time_dict["gen_proposals"] = time.time() - start
        start = time.time()

        if cfg.stage5.show_results:
            temp_v = [v for v in video_names if str(cfg.stage5.show_id) in v][0]
            # temp_v = pred_df.iloc[100]['video-id']
            # print(temp_v)
            stage45_figure(cfg, temp_v, VIDEO_DIR, FIGURE_DIR, VLM_OUTPUT_DIR, gt_df, label_map, pred_df)

    if cfg.evaluate:
        tiou_thresholds = cfg.dataset.tiou_thresholds
        activity_index = numpy.unique(list(label_map.values()))

        n_gt = len(gt_df['video-id'].unique())
        n_pred = len(pred_df['video-id'].unique())

        if n_pred < n_gt:
            logging.debug(f"# of unique predicted videos ({n_pred}) is less than # of GT videos ({n_gt}).")
            filtered_gt_df = gt_df[gt_df["video-id"].isin(pred_df["video-id"].unique())]
        else:
            filtered_gt_df = gt_df

        mAP, average_mAP = evaluate(tiou_thresholds, activity_index, filtered_gt_df, pred_df)
        time_dict["eval"] = time.time() - start
        
    if cfg.verbose:
        logging.info(time_dict)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")