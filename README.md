# Metric Distillation

## Installation

1. Create an Anaconda environment: `conda create -n metric_distillation python=3.9 -y`
2. Activate the environment: `conda activate metric_distillation`
3. Install the dependencies:
```
conda install -c conda-forge cudatoolkit=11.3.1 cudatoolkit-dev=11.3.1 cudnn=8.2 -y
pip install -r requirements.txt --no-deps
```
5. Export environment variables
```
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```


### Alternative installation

```
PIP_NO_DEPS=1 conda env create
conda activate metric_distillation
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```



## Running experiments

Check `lp_metric_distillation.py` for available tasks and specific hyperparameters. You can turn on `--debug` and `--run_tf_eagerly` to run the code in debug mode.

### Online GCRL experiments

```
python lp_metric_distillation.py 
--env_name=fetch_reach
--max_number_of_steps=500_000
--seed=0
--lp_launch_type=local_mp
--exp_log_dir=LOG_DIR
```

### Offline GCRL experiments

- CMD (default): you can tune hyperparameters as you want

```
python lp_metric_distillation.py 
--env_name=offline_ant_umaze
--max_number_of_steps=500_000
--seed=0
--lp_launch_type=local_mp
--exp_log_dir=LOG_DIR
```

- Contrastive CPC

```
python lp_metric_distillation.py 
--env_name=offline_ant_umaze
--max_number_of_steps=500_000
--seed=0
--lp_launch_type=local_mp
--exp_log_dir=LOG_DIR
--bc_coef=0.05
--bc_loss=mle
--repr_norm=false
--use_cpc=true
--contrastive_triangle_ineq=true
```

- Contrastive NCE

```
python lp_metric_distillation.py 
--env_name=offline_ant_umaze
--max_number_of_steps=500_000
--seed=0
--lp_launch_type=local_mp
--exp_log_dir=LOG_DIR
--bc_coef=0.05
--bc_loss=mle
--repr_norm=false
--use_nce=true
--contrastive_triangle_ineq=true
```

- GCBC


```
python lp_metric_distillation.py 
--env_name=offline_ant_umaze
--max_number_of_steps=500_000
--seed=0
--lp_launch_type=local_mp
--exp_log_dir=LOG_DIR
--bc_coef=0.05
--bc_loss=mle
--repr_norm=false
--use_gcbc=true
--contrastive_triangle_ineq=true
```
