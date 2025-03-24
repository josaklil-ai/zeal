"""
Helper functions for loading action localization datasets.
"""

import glob
import json
import pandas as pd
import os


def thumos_read_all_data(ann_dir, split, max_videos=-1):
    assert (
        split in ["val", "test"]
    ), "split must be either 'val' or 'test'"
    
    ann_files = glob.glob(os.path.join(ann_dir, f"*_{split}.txt"))

    ann_df = pd.DataFrame()
    for ann_file in ann_files:
        if 'Ambiguous' in ann_file:
            continue
        label = os.path.basename(ann_file).split(".")[0].split("_")[0]
        ann = pd.read_csv(ann_file, delimiter=" ", header=None, names=["video-id", "-", "t-start", "t-end"])
        ann = ann.drop(columns=["-"])
        ann["label"] = label
        ann_df = pd.concat([ann_df, ann])

    labels = sorted(ann_df["label"].unique())
    label_map = {label: i for i, label in enumerate(labels)}

    ann_df["label"] = ann_df["label"].map(label_map).astype(int)
    ann_df["video-id"] = ann_df["video-id"].astype(str)
    ann_df["t-start"] = ann_df["t-start"].astype(float)
    ann_df["t-end"] = ann_df["t-end"].astype(float)

    unique_videos = ann_df["video-id"].unique()

    assert (
        # there are 213 videos in the test set, but one video contains only "Ambiguous" class
        len(unique_videos) == 212 if split == "test" else 200
    ), "Number of videos in the dataset is not correct."

    if 0 < max_videos < len(unique_videos):
        limited_videos = unique_videos[:max_videos]
        ann_df = ann_df[ann_df["video-id"].isin(limited_videos)]
        unique_videos = limited_videos 

    return unique_videos, ann_df, label_map


def thumos_read_data_single_vid(ann_dir, split, video_name):
    pass


def thumos_read_data_single_class(ann_dir, split, action_class):
    pass


def an_read_all_data(ann_dir, split, max_videos=-1):
    assert (
        split in ["training", "validation", "testing"]
    ), "split must be either 'training', 'validation' or 'testing'"

    ann_file = os.path.join(ann_dir, 'activity_net.v1-3.min.json')
    data_dict = json.load(open(ann_file, 'r'))['database']

    ann_df = pd.DataFrame()
    for video_name, ann_dict in data_dict.items():
        if ann_dict['subset'] == split:
            for ann in ann_dict['annotations']:
                start, end = ann['segment'][0], ann['segment'][1]
                label = ann['label']
                ann_df = pd.concat([ann_df, pd.DataFrame({
                    "video-id": [f"v_{video_name}"],
                    "t-start": [start],
                    "t-end": [end],
                    "label": [label]
                })])

    labels = sorted(ann_df["label"].unique())
    label_map = {label: i for i, label in enumerate(labels)}

    ann_df["label"] = ann_df["label"].map(label_map).astype(int)
    ann_df["video-id"] = ann_df["video-id"].astype(str)
    ann_df["t-start"] = ann_df["t-start"].astype(float)
    ann_df["t-end"] = ann_df["t-end"].astype(float)

    unique_videos = ann_df["video-id"].unique()

    assert (
        len(ann_df["video-id"].unique()) == 4926 if split == "validation" else 10024
    ), "Number of videos in the dataset is not correct."

    if 0 < max_videos < len(unique_videos):
        limited_videos = unique_videos[:max_videos]
        ann_df = ann_df[ann_df["video-id"].isin(limited_videos)]
        unique_videos = limited_videos 
    
    return unique_videos, ann_df, label_map


def an_read_data_single_vid(ann_dir, split, video_name):
    pass


def an_read_data_single_class(video_dir, split, action_class):
    pass


def read_data(
    dataset, 
    ann_dir=None, 
    split=None, 
    video_name=None, 
    action_class=None,
    max_videos=-1,    
):
    if dataset == 'thumos14':
        if video_name is not None:
            raise NotImplementedError
        elif action_class is not None:
            raise NotImplementedError
        else:
            return thumos_read_all_data(ann_dir, split, max_videos)
        
    elif dataset == 'activitynetv1_3':
        split = 'validation' if split == 'test' else 'training'
        if video_name is not None:
            raise NotImplementedError
        elif action_class is not None:
            raise NotImplementedError
        else:
            return an_read_all_data(ann_dir, split, max_videos)
    

if __name__ == "__main__":
    # video_dir = "/pasteur/data/thumos14/videos"
    # ann_dir = "/pasteur/data/thumos14/annotations_test"
    video_dir = "/pasteur/data/ActivityNet200/videos"
    read_data(dataset="activitynetv1_3", ann_dir="/pasteur/u/josaklil/neurips2024/assets/", split="test")