import json
import logging
import os
import random
import sys
import numpy as np

import torch
import torch.nn.functional as F

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from main_transformer import MODEL_CLASSES, get_glue_tensor_dataset, my_evaluate

from src.general import create_dir
from src.temperature_scaling import tune_temperature

from src.transformers.processors import processors, output_modes

from sys_config import CACHE_DIR, DATA_DIR, CKPT_DIR, RES_DIR
from src.glue.run_glue import compute_metrics
from src.metrics import uncertainty_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def temp_scores_(new_logits,labels,temperature,task):
    preds = np.argmax(new_logits.detach().numpy(), axis=1)
    result = compute_metrics(task, preds, labels)
    calibration_scores = uncertainty_metrics(new_logits.detach(), labels)
    calibration_scores.update({"temperature": float(temperature)})
    result.update(calibration_scores)
    return result

def create_one_hot(n_classes):
    """Creates one-hot label tensor."""

    smoothing_value = 0.
    one_hot = torch.full((n_classes,), smoothing_value).float()
    return one_hot

def cross_entropy(output, target, n_classes):
    """
    Computes cross-entropy with KL divergence from predicted distribution
    and true distribution, specifically, the predicted log probability
    vector and the true one-hot label vector.
    """

    model_prob = create_one_hot(n_classes)
    model_prob[target] = 1. 
    return F.kl_div(output, model_prob, reduction='sum').item()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    ##########################################################################
    # Setup args
    ##########################################################################
    parser.add_argument("--local_rank",
                        type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str, help="Pretrained ckpt")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=True,
        type=bool,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true",
        default=False,
        help="Set this flag if you are using an uncased model.",
    )
    # parser.add_argument("--tapt", default=None, type=str,
    #                     help="ckpt of tapt model")
    ##########################################################################
    # Training args
    ##########################################################################
    parser.add_argument("--do_train", default=True, type=bool, help="If true do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("-seed", "--seed", required=False, type=int, help="seed")
    parser.add_argument("--indicator", required=False,
                        default=None,
                        type=str,
                        help="experiment indicator")
    parser.add_argument("-patience", "--patience", required=False, type=int, help="patience for early stopping (steps)")
    parser.add_argument("--use_adapter", required=False, type=bool,
                        default=False,
                        help="if True finetune model with added adapter layers")
    parser.add_argument("--use_bayes_adapter", required=False, type=bool,
                        default=False,
                        help="if True finetune model with added Bayes adapter layers")
    parser.add_argument("--unfreeze_adapters", required=False, type=bool,
                        default=False,
                        help="if True add adapters and fine-tune all model")
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument("--dataset_name", default=None, required=True, type=str,
                        help="Dataset [mrpc, ag_news, qnli, sst-2, trec-6]")
    # parser.add_argument("--task_name", default=None, type=str, help="Task [MRPC, AG_NEWS, QNLI, SST-2]")
    parser.add_argument("--max_seq_length", default=256, type=int, help="Max sequence length")
    ##########################################################################
    # Uncertainty estimation args
    ##########################################################################
    parser.add_argument("--unc_method",
                        default="vanilla",
                        type=str,
                        help="Choose uncertainty estimation method from "
                             "[vanilla, mc, ensemble, temp_scale, bayes_adapt, bayes_top]"
                        )
    parser.add_argument("--test_all_uncertainty", required=False, type=bool, default=True,
                        help=" if True evaluate [vanilla, mc_3, mc_5, mc_10, mc_20, temp_scaling] "
                             "uncertainty methods for the model")
    parser.add_argument("--bayes_output", required=False, type=bool, default=False,
                        help=" if True add Bayesian classification layer (UA)")
    ##########################################################################
    # Server args
    ##########################################################################
    parser.add_argument("-g", "--gpu", required=False,
                        default='0', help="gpu on which this experiment runs")
    parser.add_argument("-server", "--server", required=True,
                        default='ford', help="server on which this experiment runs")

    args = parser.parse_args()

    # Setup
    if args.server is 'ford':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("\nThis experiment runs on gpu {}...\n".format(args.gpu))
        args.n_gpu = 1
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.n_gpu = 0 if args.no_cuda else 1
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1

    print('device: {}'.format(args.device))

    #########################################
    # Setup args
    #########################################
    if args.seed == None:
        seed = random.randint(1, 9999)
        args.seed = seed

    args.task_name = args.dataset_name.upper()
    args.cache_dir = CACHE_DIR
    args.data_dir = os.path.join(DATA_DIR, args.task_name)
    if args.dataset_name == 'cola': args.data_dir = os.path.join(DATA_DIR, "CoLA")
    args.overwrite_cache = True
    args.evaluate_during_training = True

    # Output dir
    output_dir = os.path.join(CKPT_DIR, '{}_{}'.format(args.dataset_name, args.model_type))
    args.output_dir = os.path.join(output_dir, 'all_{}'.format(args.seed))
    if args.use_adapter: args.output_dir += '-adapter'
    if args.use_bayes_adapter: args.output_dir += '-bayes-adapter'
    if args.indicator is not None: args.output_dir += '-{}'.format(args.indicator)
    if args.patience is not None: args.output_dir += '-early{}'.format(int(args.num_train_epochs))
    if args.bayes_output: args.output_dir += '-bayes-output'
    if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: args.output_dir += '-unfreeze'
    args.current_output_dir = args.output_dir
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):

        print('No model here! {}'.format(args.output_dir))
        exit()
    #########################################
    # Setup logging
    #########################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    #########################################
    # Prepare GLUE task
    #########################################
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_classes = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    
    path = os.path.join(RES_DIR, '{}_{}_100%'.format(args.task_name, args.model_type))
    create_dir(path)
    name = 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(args.seed, args.learning_rate,
                                                  args.per_gpu_train_batch_size,
                                                  int(args.num_train_epochs))
    # if args.use_adapter: name += '_adapters'
    if args.indicator is not None: name += '_{}'.format(args.indicator)
    print(name)

    dirname = os.path.join(path, name)

    # First check in domain
    print('Evaluate temperature scaling....')
    filename = 'temp_scale_results_NEW'
    if args.use_adapter: filename += '_adapter'
    if args.use_bayes_adapter: filename += '_bayes_adapter'
    if args.bayes_output: filename += '_bayes_output'
    if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
    temp_json_file_id = os.path.join(dirname, '{}.json'.format(filename))
    temp_json_file_ood = os.path.join(dirname, '{}_ood.json'.format(filename))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = model_class.from_pretrained(args.output_dir)
    model.to(args.device)

    #########################################
    # Check if experiment already done
    #########################################
    # path = os.path.join(RES_DIR, '{}_{}_100%'.format(args.task_name, args.model_type))
    # create_dir(path)
    # name = 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(args.seed, args.learning_rate,
    #                                               args.per_gpu_train_batch_size,
    #                                               int(args.num_train_epochs))
    # # if args.use_adapter: name += '_adapters'
    # if args.indicator is not None: name += '_{}'.format(args.indicator)
    # print(name)

    # dirname = os.path.join(path, name)

    #######################
    # Test uncertainty OOD
    #######################
    if args.dataset_name == 'mnli':
        test_dataset_ood = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True, ood=True)
    elif args.dataset_name == 'qqp':
        # test_dataset_ood = get_glue_tensor_dataset(None, args, 'mrpc', tokenizer, test=True, data_dir=os.path.join(DATA_DIR, 'MRPC'))
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'twitterppdb', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'TwitterPPDB'))
    elif args.dataset_name == 'mrpc':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'qqp', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'QQP'))
    elif args.dataset_name == 'sst-2':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'imdb', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'IMDB'))
    elif args.dataset_name == 'imdb':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'sst-2', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'SST-2'))
    elif args.dataset_name == 'rte':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'qnli', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'QNLI'))
    elif args.dataset_name == 'qnli':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'rte', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'RTE'))
    else:
        test_dataset_ood = None
        # pass #raise NotImplementedError

    
    # Temperature Scaling
    
    # ID temperature scaling
    eval_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, evaluate=True)
    test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True)

    vanilla_results_val, vanilla_val_logits = my_evaluate(eval_dataset, args, model, mc_samples=None)
    vanilla_results_test, vanilla_test_logits = my_evaluate(test_dataset, args, model, mc_samples=None)

    best_nll = float('inf')
    best_temp = -1

    temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

    n_classes = len(set(vanilla_results_val['gold_labels']))

    for temp in temp_values:
        nll = np.mean(
            [
                cross_entropy(
                    F.log_softmax(vanilla_val_logits[_i_] / temp, 0), vanilla_results_val['gold_labels'][_i_], n_classes
                )
                for _i_ in range(len(vanilla_val_logits))
            ]
        )
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
    print(best_temp)

    new_val_logits = vanilla_val_logits / best_temp #.unsqueeze(1).expand(vanilla_val_logits.size(0), vanilla_val_logits.size(1))
    new_test_logits = vanilla_test_logits / best_temp #.unsqueeze(1).expand(vanilla_test_logits.size(0), vanilla_test_logits.size(1))


    temp_scores_val = temp_scores_(new_val_logits,vanilla_results_val['gold_labels'],best_temp,args.dataset_name)
    temp_scores_test = temp_scores_(new_test_logits,vanilla_results_test['gold_labels'],best_temp,args.dataset_name)

    temp_scores = {"val_results": temp_scores_val, "test_result": temp_scores_test}
    
    with open(temp_json_file_id, 'w') as f:
        json.dump(temp_scores, f)
    
    if test_dataset_ood is not None:
        vanilla_results_test_ood, vanilla_test_logits_ood = my_evaluate(test_dataset_ood, args, model, mc_samples=None)
        new_logits_ood = vanilla_test_logits_ood / best_temp #.unsqueeze(1).expand(vanilla_test_logits_ood.size(0), vanilla_test_logits_ood.size(1))
        
        temp_scores_ood = temp_scores_(new_logits_ood,vanilla_results_test_ood['gold_labels'],best_temp,args.dataset_name)
        with open(temp_json_file_ood, 'w') as f:
            json.dump(temp_scores_ood, f)
