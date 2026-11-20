# MoE-FFD Smoke Test Dataset

## Overview
This folder contains the sample testing dataset prepared for
the baseline and modified MoE-FFD model smoke-test experiments.

## Dataset Information
- Source dataset: Celeb-DF
- Total testing videos: 20
- Real videos: 10
- Fake videos: 10
- Total extracted frames: 320
- Frames per video: 16
- Approximate folder size: 52 MB

## Folder Structure

testingdataset/
├── README.md
├── videos/
│   ├── Celeb-real/
│   └── Celeb-synthesis/
└── frames/
    ├── Celeb-real/
    └── Celeb-synthesis/

## Preparation Process
1. Selected 10 real and 10 fake videos from the Celeb-DF collection.
2. Organized the selected videos into their respective categories.
3. Extracted 16 frames from each video.
4. Stored the extracted frames in corresponding video-named folders.
5. Used the same sample data to test both baseline and modified models.

## Purpose
The testing dataset is used for:
- Verifying model initialization and checkpoint loading.
- Testing video frame extraction and inference.
- Comparing baseline and modified model predictions.
- Evaluating frame-level and video-level performance.

## Important Note
This is a small smoke-test sample and is not intended to represent
full-dataset model performance.

The original Celeb-DF dataset and its derived files are subject
to the dataset authors' access and usage terms. Redistribution
requires appropriate permission.
