## flow-diffusion  

This repository contains the code for training video policies presented in our work [Learning to Act from Actionless Video through Dense Correspondences](https://flow-diffusion.github.io/). 

[website](https://flow-diffusion.github.io/) | [arXiv]() | [paper]()

https://github.com/flow-diffusion/flow-diffusion/assets/43379407/065ef3b2-44e8-4af0-8c26-a70e6d080aed


## Getting started  

We recommend to create a new environment with pytorch installed using conda.   

```bash  
conda create -n avdc python=3.9
conda activate avdc
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```  

Next, install the requirements  

```bash
pip install -r requirements.txt
```


## Dataset structure

We contain example dataset structure in `datasets/`.   

The pytorch dataset and dataloaders are defined in `flowdiffusion/datasets.py`


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

To resume training, you can use `-c --checkpoint_num` argument.  
```bash
# This will resume training with 1st checkpoint (should be named as model-1.pt)
python train_mw.py --mode train --c 1
```


## Inferencing

Use the following arguments for inference  
`-p --inference_path`: specify input image path  
`-t --text`: specify the text discription of task  

For example:  
```bash
python train_mw.py --mode inference -c 1 -p ../examples/assembly.png -t assembly
```


## Acknowledgements

This codebase is modified from the following repositories:  
[imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)  
[guided-diffusion](https://github.com/openai/guided-diffusion)  
