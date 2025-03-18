# Pi0 and Lerobot with Rerun
A repo to explore training robots with Pi0 and Lerobot and human pose motion retargeting

<p align="center">
  <a title="Website" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
          <img src="https://img.shields.io/badge/Rerun-0.21.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
      </a>
    <a title="Website" href="https://www.physicalintelligence.company/blog/openpi" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    <a title="arXiv" href="https://arxiv.org/html/2410.24164v1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    <a title="Github" href="https://github.com/pablovela5620/pi0-lerobot" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/prompt-da?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
  </p>

<p align="center">
  <img src="media/lerobot_notebooks.gif" alt="example output" width="720" />
</p>

## Installation
Currently only linux is supported
### Using Pixi
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/rerun-io/pi0-lerobot.git
cd pi0-lerobot
```

## Usage
### Human Pose and Kinematics

<p align="center">
  <img src="media/assembly101.gif" alt="example output" width="640" />
</p>

For HOCap dataset, sample is automatically downloaded from dataset on command run. Checkout example rrd file [here](https://app.rerun.io/version/0.22.1/index.html?url=https://huggingface.co/datasets/pablovela5620/rrd-examples/resolve/main/hocap-example.rrd)

For the full assembly101 dataset go to this [link](https://github.com/assembly-101/assembly101-download-scripts) to get/download dataset

To run dataset visualization
```bash
pixi run visualize-hocap-dataset # recommended
pixi run visualize-assembly101-dataset
```

Current Pipeline
<p align="center">
  <img src="media/multiview_keypoint_tracker.png" alt="pipeline" />
</p>


To run 2D pose estimation, tracking, and triangulation
```bash
pixi run pose-estimation-assembly101
```
### Jupyter Notebook Tutorials
```bash
pixi run notebook_tutorial
```

## TODO
### Part 1. Human Pose and Kinematics
- [x] Basic Triangulation from 2D Detection (body pose)
- [x] Basic Triangulation from 2D detection (hand pose)
- [x] Detection by Tracking (extrapolate 3d views-> Generate bounding box based on extrapolated -> check based on kpts confidence)
- [ ] Mano + SMPL fitting for skeleton kinematics
- [ ] Add EgoCentric (first person) headset views and poses

### Part 2. Body and Hand Pose Retargeting
- [ ] Implement pose retargeting to isaac sim

### Part 3. Immitation Learning and Teleoperation
- [x] Notebook Tutorials
- [ ] Finetune and evaluation on Aloha simulator with Pi0
- [ ] Finetune base Pi0 on [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) from lerobot
