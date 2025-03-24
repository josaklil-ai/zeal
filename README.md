# ZEAL: Zero-shot Action Localization via the Confidence of Large Vision-Language Models 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2410.14340)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Codebase for the paper introducing [ZEAL](https://arxiv.org/pdf/2410.14340), a novel method for zero-shot action localization in long-form videos leveraging vision-language models.

### Setup
```bash
conda create -n zeal python=3.12
conda activate zeal
pip3 install -r requirements.txt
```
Create a `.env` file in the project root directory and fill in your `OPENAI_API_KEY`, along with an `OUTPUT_DIR` and `LOG_DIR` for storing outputs.

### Usage
This repo allows modular usage of the ZEAL pipeline. We also provide example scripts for running stages 2 and 4 on multiple GPUs for SLURM managed clusters in `scripts/`.

#### Stage 1: Generate VLM Queries
To generate yes/no questions for each action class with a given LLM, run
```bash
python -m run stage1.enable=True stage1.llm_type=<LLM_TYPE>
```
This will generate a `message_{LLM_TYPE}.json` file in the `llm_output` directory with start, end, and description queries for each action class. You only need to run this once.

***
#### Stage 2: Generate CLIP-actionness scores
To compute CLIP similarity scores between video frames and action descriptions, run
```bash
python -m run stage2.enable=True stage2.clip_model.arch=<CLIP_ARCH>
```
This will generate CLIP features for each video frame (at the fps specified `stage4.fps`) in the `clip_output` directory. You only need to run this once.

***
#### Stage 3: Filter action classes
To filter action classes based on `stage3.n_frames` frames sampled from each video, run
```bash
python -m run stage3.enable=True stage3.use_gt=False
```
This will generate a multi-hot vector for each video indicating candidate action classes for the video. If `stage3.use_gt` is set to `True`, the ground truth action classes will be used instead of the candidate action classes. You only need to run this once.

***
#### Stage 4: Generate VLM confidence scores (ZEAL)
To generate soft confidence scores for video frames at `stage4.fps` for fine-grained localization, run
```bash
python -m run stage3.enable=True stage3.use_gt=<True/False> stage4.enable=True stage4.vlm_type=<VLM_TYPE>
```
This will generate scores for each video frame and action class in the `vlm_output` directory. Toggle `stage3.use_gt=True` to use the ground truth action classes instead of the candidate action classes.

***
#### Stage 5: Generate proposals
To generate final action localization proposals for each video, run
```bash
python -m run stage3.use_gt=<True/False> stage5.enable=True
```

***
### Evaluation
To evaluate the performance of ZEAL on a dataset (using or not using ground truth action classes), run
```bash
python -m run dataset=<DATASET> stage3.use_gt=<True/False> evaluate=True
```

### Citation
If you find our work useful, please consider citing our paper:
```bibtex
@misc{aklilu2024ZEAL,
    title={Zero-shot Action Localization via the Confidence of Large Vision-Language Models}, 
    author={Josiah Aklilu and Xiaohan Wang and Serena Yeung-Levy},
    year={2024},
    eprint={2410.14340},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.14340}, 
}
```