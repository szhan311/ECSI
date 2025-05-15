This repo builds on top of the official [DDBM](https://github.com/alexzhou907/DDBM)

# Installation
## Download DDBM pretrained checkpoints from [hugging face](https://huggingface.co/alexzhou907/DDBM)

| Description                                         | File                                      |
|-----------------------------------------------------|-------------------------------------------|
|                   DIODE checkpoint                  |           ddbm_diode_vp_ema.pt            |


Save checkpoints at folder weights/diode_ema_0.9999_440000.pt



## dataset
We provided demo images for sampling. For full dataset, , please download from [here](https://diode-dataset.org/)

## env

```
conda create -n ecsi python=3.10
conda activate ecsi
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
pip install k-diffusion
conda install -c conda-forge mpi4py openmpi
pip install -e .
```

# Usage
```
sh sample_diode.sh
```
