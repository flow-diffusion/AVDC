## AVDC

The official codebase for training video policies in AVDC

https://github.com/flow-diffusion/flow-diffusion.github.io/assets/43379407/9aa380df-0ff7-4c41-af2d-d67d23c53e72

This repository contains the code for training video policies presented in our work   
[Learning to Act from Actionless Video through Dense Correspondences](https://flow-diffusion.github.io/)  
[Po-Chen Ko](https://pochen-ko.github.io/),
[Jiayuan Mao](https://jiayuanm.com/),
[Yilun Du](https://yilundu.github.io/),
[Shao-Hua Sun](https://shaohua0116.github.io/),
[Joshua B. Tenenbaum](https://cocosci.mit.edu/josh)  
[website](https://flow-diffusion.github.io/) | [paper](https://flow-diffusion.github.io/AVDC.pdf)| [arXiv](https://arxiv.org/abs/2310.08576)

```bib
@article{Ko2023Learning,
  title={{Learning to Act from Actionless Video through Dense Correspondences}},
  author={Ko, Po-Chen and Mao, Jiayuan and Du, Yilun and Sun, Shao-Hua and Tenenbaum, Joshua B},
  journal={arXiv preprint},
  year={2023},
}
```


## Getting started  

We recommend to create a new environment with pytorch installed using conda.   

```bash  
conda create -n avdc python=3.9
conda activate avdc
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```  

Next, clone the repository and install the requirements  

```bash
git clone https://github.com/flow-diffusion/AVDC
cd AVDC
pip install -r requirements.txt
```


## Dataset structure

This repo contains example dataset structure in `datasets/`.   

The pytorch dataset classes are defined in `flowdiffusion/datasets.py`


## Training models

For Meta-World experiments, run
```bash
cd flowdiffusion
python train_mw.py --mode train
# or python train_mw.py -m train
```

or run with `accelerate`
```bash
accelerate launch train_mw.py
```

For iTHOR experiments, run `train_thor.py` instead of `train_mw.py`  
For bridge experiments, run `train_bridge.py` instead of `train_mw.py`  

The trained model should be saved in `../results` folder  

To resume training, you can use `-c` `--checkpoint_num` argument.  
```bash
# This will resume training with 1st checkpoint (should be named as model-1.pt)
python train_mw.py --mode train -c 1
```

## Inferencing

Use the following arguments for inference  
`-p` `--inference_path`: specify input image path  
`-t` `--text`: specify the text discription of task  

For example:  
```bash
python train_mw.py --mode inference -c 1 -p ../examples/assembly.png -t assembly
```

## Pretrained models 

We also provide checkpoints of the models described in our experiments as following.   
[Meta-World](https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt) |  [iTHOR](https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-30.pt) | [Bridge](https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/bridge/model-42.pt)   

Download and put the .pt file in `results/[environment]` folder. The resulting directory structure should be `results/{mw, thor, bridge}/model-[x].pt`, for example `results/mw/model-24.pt`

Or use `download.sh`
```bash
./download.sh metaworld
# ./download.sh ithor
# ./download.sh bridge
```

After this, you can use argument `-c [x]` to resume training or inference with our checkpoint. For example:  
```bash
python train_mw.py --mode train -c 24
```
Or  
```bash
python train_mw.py --mode inference -c 24 -p ../examples/assembly.png -t assembly
```



## Acknowledgements

This codebase is modified from the following repositories:  
[imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)  
[guided-diffusion](https://github.com/openai/guided-diffusion)  
