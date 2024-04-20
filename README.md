# A Closer Look at Spatial-Slice Features Learning for COVID-19 Detection


### E2D [[Paper Link]](https://arxiv.org/abs/2404.01643) [[Project Page]](https://allproj001.github.io/cov19d.github.io/)

[Chih-Chung Hsu](https://cchsu.info/), [Chia-Ming Lee](https://ming053l.github.io/), [Yi-Shiuan Chou](https://scholar.google.com/citations?&user=iGX8FBcAAAAJ)

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

<img src=".\figures\spatialslice.png" width="500"/>

<img src=".\figures\kde.png" width="600"/>

<img src=".\figures\generalize.png" width="400"/>

## Environment
### Installation
```
git clone https://github.com/ming053l/E2D.git
conda create --name e2d python=3.8 -y
conda activate e2d
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
## How To Test

- Run `./preproceesing/inference/*` step by step and then (you need to change directory within all file.)
```
CUDA_VISIBLE_DEVICES=0,1 python inference.py
```

## How To Train
- Run `./preproceesing/*` step by step and then (you need to change directory within all file.)
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## Citations
#### BibTeX
    @misc{hsu2024closer,
        title={A Closer Look at Spatial-Slice Features Learning for COVID-19 Detection}, 
        author={Chih-Chung Hsu and Chia-Ming Lee and Yang Fan Chiang and Yi-Shiuan Chou and Chih-Yu Jiang and Shen-Chieh Tai and Chi-Han Tsai},
        year={2024},
        eprint={2404.01643},
        archivePrefix={arXiv},
        primaryClass={eess.IV}
    }
    @INPROCEEDINGS{10192945,
        author={Chih-Chung Hsu and Chia-Ming Lee and Yang Fan Chiang and Yi-Shiuan Chou and Chih-Yu Jiang and Shen-Chieh Tai and Chi-Han Tsai},
        booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
        title={Bag of Tricks of Hybrid Network for Covid-19 Detection of CT Scans}, 
        year={2023},
        pages={1-4}
        }


## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.
