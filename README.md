# transformer-uncertainty
Code for evaluating uncertainty estimation methods for Transformer-based architectures in natural language understanding tasks.

## Uncertainty Methods
* Vanilla (deterministic)
* Monte Carlo dropout
* Temperature scaling
* Ensembles
* Bayesian Layer (final layer or adapter layer)

## Environment
Create a conda environment, download the required torch package and the rest of the requirements:
```
conda create -n unc_env python=3.7
conda activate unc_env
conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```
See this [pytorch page](https://pytorch.org/get-started/previous-versions/) to download correctly pytorch to your machine.

## Datasets
Download GLUE datasets:
```
bash get_data.sh
```
