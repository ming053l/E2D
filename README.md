# A Closer Look at Spatial-Slice Features Learning for COVID-19 Detection


### SSFL++, E2D [[Paper Link]](https://arxiv.org/abs/2404.01643) [[Project Page]](https://allproj001.github.io/cov19d.github.io/) [[Technical Report]](https://arxiv.org/abs/2403.11230)

[Chih-Chung Hsu](https://cchsu.info/), [Chia-Ming Lee](https://ming053l.github.io/), Yang Fan Chiang, [Yi-Shiuan Chou](https://scholar.google.com/citations?&user=iGX8FBcAAAAJ), Chih-Yu Jiang, Sheng-Chieh Tai, Chi-Han Tsai

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

- SSFL++: Remove redundancy on spatial/slice dimension with kernel-density-aware slice sampling for CT scan. (removing 70% reduncancy and with global sequantial modeling)

- E2D: Simple 2D-CNN for COVID-19 Detection. (few-shot aware, robust, and efficient)
<img src=".\figures\spatialslice.png" width="500"/>
<img src=".\figures\kde.png" width="600"/>

- Generalizability: A Unified redundancy removal framework for CT-scan-like data.

<img src=".\figures\generalize.png" width="600"/>

# Data Reduction Overview

The reduction in redundant data achieved by the SSFL++ module is evaluated across three dimensions: spatial, slice, and overall. This approach quantifies the efficiency of the SSFL++ module in reducing unnecessary information in CT scans, enabling more focused analysis and processing. By minimizing data redundancy, the module enhances computational efficiency and potentially improves the accuracy of subsequent analyses or models applied to the CT data.

|              | Spatial Area (K) |          |          | Slice Length |          |          | Spatial × Slice (M) |        |         |
|--------------|------------------|----------|----------|--------------|----------|----------|---------------------|--------|---------|
|              | Before           | After    | Δ (%)    | Before       | After    | Δ (%)    | Before              | After  | Δ (%)   |
| Training     | 267.25           | 155.53   | 0.4184   | 285.32       | 142.91   | 0.4983   | 76.25               | 22.22  | 0.7085  |
| Positive     | 266.42           | 157.69   | 0.4088   | 295.90       | 148.18   | 0.4985   | 78.83               | 23.36  | 0.7036  |
| Negative     | 268.21           | 153.03   | 0.4296   | 273.97       | 137.26   | 0.4981   | 73.48               | 21.00  | 0.7141  |
| Validation   | 265.62           | 155.23   | 0.4172   | 281.95       | 141.23   | 0.4984   | 74.89               | 21.92  | 0.7072  |
| Positive     | 268.94           | 160.48   | 0.4061   | 280.53       | 140.55   | 0.4984   | 75.45               | 22.55  | 0.7010  |
| Negative     | 262.12           | 149.69   | 0.4288   | 283.49       | 141.97   | 0.4984   | 74.30               | 21.25  | 0.7139  |
| (T+V) Positive | 267.25         | 155.53   | 0.4184   | 292.96       | 146.72   | 0.4985   | 78.29               | 22.81  | 0.7085  |
| (T+V) Negative | 267.01         | 152.37   | 0.4294   | 275.78       | 138.16   | 0.4982   | 73.64               | 21.05  | 0.7141  |
| Total         | 266.94          | 155.47   | 0.4182   | 284.68       | 142.59   | 0.4983   | 75.99               | 22.16  | 0.7082  |
| Testing       | 279.55          | 153.41   | 0.4520   | 309.39       | 154.67   | 0.5003   | 86.48               | 23.72  | 0.7256  |

*Table: The reduction metrics by the SSFL++ module across training/valid/testing set.*


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
    @misc{hsu2024simple,
      title={Simple 2D Convolutional Neural Network-based Approach for COVID-19 Detection}, 
      author={Chih-Chung Hsu and Chia-Ming Lee and Yang Fan Chiang and Yi-Shiuan Chou and Chih-Yu Jiang and Shen-Chieh Tai and Chi-Han Tsai},
      year={2024},
      eprint={2403.11230},
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
    @InProceedings{Hsu_2024_CVPR,
        author    = {Hsu, Chih-Chung and Lee, Chia-Ming and Chiang, Yang Fan and Chou, Yi-Shiuan and Jiang, Chih-Yu and Tai, Shen-Chieh and Tsai, Chi-Han},
        title     = {A Closer Look at Spatial-Slice Features Learning for COVID-19 Detection},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2024},
        pages     = {4924-4934}
    }


## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.
