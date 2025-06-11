# THGD: Topology-Aware Hierarchical Graph Diffusion Model  
This implementation is based on [DiGress](https://github.com/cvignac/DiGress)'s excellent work. 

## Environment Setup  
The code has been tested with:  
- PyTorch 2.2  
- CUDA 11.8  
- PyTorch Geometric 2.3.1  

```bash
conda create -n THGD python=3.10 -y
pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter==2.1.2
pip install -e .
```

## Running the Code  
All executions are launched via `python3 main.py`. Refer to [Hydra documentation](https://hydra.cc/) for parameter customization.

### Training  
The model training consists of two sequential phases:  
1. **Coarse Model Training** (learns high-level molecular topologies)  
2. **Refinement Model Training** (recovers atomic-level details)  

Pre-configured training files are provided in `configs/`. Example for GuacaMol:  

1. Train coarse model:  
```bash
cd src
python main.py +guacamol_exp=coarse
```

2. Train refinement model:  
```bash
python main.py +guacamol_exp=refine
```

**Notes**:  
- Dataset preprocessing (feature extraction for coarse/expanded graphs) runs on CPU and typically takes ~30 minutes (device-dependent). It is recommended to use our preprocessed dataset here:  
    - https://drive.google.com/file/d/1Rr5EOMLNOdOYCj6L5VzQaS95NooPDbyC/view?usp=sharing
- This preprocessing must complete before training begins.  

### Sampling  
Sampling requires:  
1. Trained coarse model checkpoint  
2. Trained refinement model checkpoint  
3. Precomputed optimal prior distribution tensor (obtain by preprocessing)  

#### De Novo Generation  
Use pre-configured sampling profiles:  
```bash
python sample.py sample=guacamol
```

❗ **Configuration Alignment**:  
Ensure the checkpoint's training config matches your sampling config (e.g., `coarse_cfg` in YAML must correspond to the checkpoint's original training config).

#### Scaffold-Constrained Generation  
Modify the `scaffold` field in the sampling config.

## Pre-trained Checkpoints  
Available checkpoint for three datasets (place downloaded files and place in `checkpoints/` folder): 

https://drive.google.com/file/d/1nOrOq6Jf7adqG0vbitnbFPJ9_-AzdHcX/view?usp=sharing

| Dataset    | Coarse Model (NLL) | Refinement Model (NLL) |  
|------------|---------------------|------------------------|  
| ZINC250k   | 44.9         | 82.9            |  
| MOSES      | 37.5         | 72.0            |  
| GuacaMol   | 59.8         | 94.6            |  

## Generated Samples  
We provide few example outputs in `sample-results/`.