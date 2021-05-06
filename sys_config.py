import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

glue_datasets = ['sst-2', 'mrpc', 'qnli', "cola", "mnli", "mnli-mm", "sts-b", "qqp", "rte", "wnli"]
available_datasets = glue_datasets + ["ag_news", "trec-6", "dbpedia", "imdb", "pubmed"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = os.path.join(BASE_DIR, 'data')

CONTR_DATA_DIR = os.path.join(DATA_DIR, 'contrast-sets')

IMDB_CONTR_DATA_DIR = os.path.join(CONTR_DATA_DIR, 'IMDb', 'data')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# RES_DIR = os.path.join(BASE_DIR, 'results')
# RES_DIR = os.path.join(BASE_DIR, 'resultsss')
RES_DIR = os.path.join(BASE_DIR, 'dokimi')

AL_RES_DIR = os.path.join(BASE_DIR, 'resultsss_al')

