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
## Run model
For example run:
```
python main_transformer --dataset_name rte 
                        --model_type bert 
                        --model_name_or_path bert-base-uncased 
                        --do_lower_case 
                        --seed 1234 
                        --unc_method bayes_adapt
```
See [main_transformer.py](https://github.com/mourga/transformer-uncertainty/blob/main/main_transformer.py) for the rest of arguments.
