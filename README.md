
<div align="center">    
 
# SILT: Shadow-aware Iterative Label Tuning for Learning to Detect Shadows from Noisy Labels 

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/ICCV-2023-4b44ce.svg)](https://www.nature.com/articles/nature14539)

![image](https://github.com/Cralence/SILT/blob/main/assets/NL_pipeline.png)

<!--  
Conference   
-->   
</div>
 
## Description   
This is the pytorch implementation of  the ICCV 2023 paper "SILT: Shadow-aware Iterative Label Tuning for Learning to 
Detect Shadows from Noisy Labels" by Han Yang, Tianyu Wang, Xiaowei Hu and Chi-Wing Fu.


## How to Run   
1. Install dependencies   
```bash
# clone project   
git clone https://github.com/Cralence/SILT.git

# create conda environment
cd SILT
conda env create -f environment.yaml
conda activate silt
pip install opencv-python
pip install omegaconf==2.3.0
 ```   

2. Download the pretrained weights for the backbone encoders from here, and set the correct path
in `configs/silt_training_config.yaml`.

3. Train the model by running:
```bash
python train.py --dataset SBU --backbone PVT-b5
```

4. Test the model by running:
```bash
python infer.py --dataset SBU --ckpt path_to_weight  
```

## Dataset
Our relabeled SBU test set can be downloaded from here.

## Pretrained Model
Our pretrained weights can be downloaded from here. 

### Citation   
```
@article{yang2023silt,
  title={SILT: Shadow-aware Iterative Label Tuning for Learning to Detect Shadows from Noisy Labels},
  author={Han Yang, Tianyu Wang, Xiaowei Hu, Chi-Wing Fu},
  journal={arXiv preprint arXiv:...},
  year={2023}
}
```   