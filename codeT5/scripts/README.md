# Scripts optimized for Accelerate

## Get started

Use one of the following steps:

1. Use Docker to load the Torch with CUDA 12.4 image

2. Manually set up the environment and install all the dependencies 

## Install dependencies

Run the following command in terminal/bash:

```bash
pip3 install -r requirements.txt
```

### This might cause issues with deepspeed. So there are a number of steps which worked for me:

**First make sure to have c/c++ compiler, python >= 3.10 and nvidia driver installed on Linux.**

1- First install **cuda12.4** and run check:
```bash
nvcc --version
```

2- Install **torch** :
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

-OR-

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

3- Install **trl** :
```bash
pip3 install trl
```

4- Install **trl** :
```bash
pip3 install trl
```

5- Install **deepspeed** :
```bash
pip3 install deepspeed
```

make sure to install everything without any errors and warnings!

## Setup the accelerator

This dir has `default_config.yaml` which is tested by me and If you want no struggle then place it in `~/.cache/huggingface/accelerate/default_config.yaml` 
-OR-

RUN:
```bash
accelerate config
```
and set all the parameters by yourself. Follow this [Image](./config.png).

## Run the script

Run the script either using:
```bash
accelerate launch trainer.py
```
OR
```bash
torchrun trainer.py --params ...
```

Where both works the same but torchrun dosent follow `default_config.yaml` so pass the params manuelly

## Params

all the params are [here](./default_config.yaml).

Use the ``Distribute Multi GPU`` for kaggle `2xT4` .

## Script

**[trainer](./trainer.py) can fine-tune any Seq2Seq Model.**
Change the model name in:
```python
base_model = "{model_name}"
```

This script is build for both local and kaggle environment.