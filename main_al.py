import glob
import json
import logging
import os
import pickle
import shutil
import sys
import time
from math import ceil

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.temperature_scaling import tune_temperature
from main_transformer import get_glue_dataset, get_glue_tensor_dataset, MODEL_CLASSES, train_transformer, my_evaluate
from src.active_learning.contrast_set import contrast_acc_imdb
from src.active_learning.uncertainty_acquisition import calculate_uncertainty
from src.general import create_dir
from src.transformers import processors
from src.transformers.processors import output_modes

from sys_config import CACHE_DIR, DATA_DIR, CKPT_DIR, IMDB_CONTR_DATA_DIR, AL_RES_DIR

logger = logging.getLogger(__name__)


def train_transformer_model(args, X_inds, X_val_inds=None,
                            iteration=None, val_acc_previous=None,
                            eval_dataset=None):
    """
        Train a transformer model for an AL iteration
    :param args: arguments
    :param X_inds: indices of original training dataset used for training
    :param X_val_inds: indices of original validation dataset used during training
    :param iteration: current AL iteration
    :param val_acc_previous: accuracy of previous AL iteration
    :param eval_dataset: ?
    :return:
    """
    if iteration is not None:
        create_dir(args.output_dir)
        args.current_output_dir = os.path.join(args.output_dir, 'iter-{}'.format(iteration))
        args.previous_output_dir = os.path.join(args.output_dir, 'iter-{}'.format(iteration - 1))

    args.task_name = args.task_name.lower()
    if args.task_name not in processors.processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors.processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        # use_adapter=args.use_adapter,
        # use_bayes_adapter=args.use_bayes_adapter,
        # adapter_initializer_range=0.0002 if args.indicator == 'identity_init' else 1,
        bayes_output=args.bayes_output,
        # unfreeze_adapters=args.unfreeze_adapters

    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("Training/evaluation parameters %s", args)

    minibatch = int(len(X_inds) / (args.per_gpu_train_batch_size * max(1, args.n_gpu)))
    args.logging_steps = min(int(minibatch / 5), 500)
    if args.logging_steps < 1:
        args.logging_steps = 1
    if args.server == 'ford':
        args.logging_steps = int(minibatch / 2) + 1

    # convert to tensor dataset
    train_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
    assert len(train_dataset) == len(X_inds)

    if eval_dataset is None:
        eval_dataset = get_glue_tensor_dataset(X_val_inds, args, args.task_name, tokenizer, evaluate=True)

    times_trained = 0
    val_acc_current = 0
    if val_acc_previous is None:
        val_acc_previous = 0.2
    val_acc_list = []
    results_list = []
    train_loss_list = []
    val_loss_list = []
    original_output_dir = args.current_output_dir

    while val_acc_current < val_acc_previous - 0.1 and times_trained < 2:
        times_trained += 1
        args.model_type = args.model_type.lower()
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.current_output_dir = original_output_dir + '_trial{}'.format(times_trained)

        model.to(args.device)
        # Train
        model, train_loss, val_loss, results = train_transformer(args, train_dataset,
                                                                 eval_dataset,
                                                                 model, tokenizer)
        accuracy = results['acc']

        val_acc_current = accuracy
        if args.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass
        val_acc_list.append((times_trained, val_acc_current))
        results_list.append(results)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    best_trial = max(val_acc_list, key=lambda item: item[1])[0]
    train_loss = train_loss_list[best_trial - 1]
    results = results_list[best_trial - 1]
    best_model_ckpt = original_output_dir + '_trial{}'.format(best_trial)
    # model = AutoModelForSequenceClassification.from_pretrained(best_model_ckpt)
    model = model_class.from_pretrained(best_model_ckpt)
    model.to(args.device)
    if os.path.isdir(original_output_dir):
        shutil.rmtree(original_output_dir)
    os.rename(best_model_ckpt, original_output_dir)
    args.current_output_dir = original_output_dir

    # Results
    train_results = {'model': model, 'train_loss': round(train_loss, 4), 'times_trained': times_trained}
    train_results.update(results)

    if train_results['acc'] > args.acc_best:
        args.acc_best_iteration = iteration
        args.acc_best = train_results['acc']
        args.best_output_dir = args.current_output_dir

    iteration_dirs = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/", recursive=True)))
    for dir in iteration_dirs:
        if dir not in [args.current_output_dir, args.best_output_dir, args.output_dir]:
            shutil.rmtree(dir)

    return train_results


def test_transformer_model(args, X_inds, model=None, ckpt=None, dataset=None):
    """
    Test transformer model on Dpool during an AL iteration
    :param args: arguments
    :param X_inds: indices of original *train* set
    :param model: model used for evaluation
    :param ckpt: path to model checkpoint
    :return:
    """
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if dataset is None:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        #     cache_dir=args.cache_dir,
        #     use_fast=args.use_fast_tokenizer,
        # )

        # it is not the eval dataset it's the Dpool
        dpool_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)

    else:
        # dataset to test model on
        dpool_dataset = dataset
    if model is None:
        model = model_class.from_pretrained(ckpt)
        model.to(args.device)
    print('MC samples N={}'.format(args.mc_samples))
    result, logits = my_evaluate(dpool_dataset, args, model, mc_samples=args.mc_samples)
    eval_loss = result['loss']
    return eval_loss, logits, result


def loop(args):
    """
    Main script for active learning algorithm.
    :param args: contains necessary arguments for model, training, data and AL settings
    :return:
    Datasets (lists): X_train_original, y_train_original, X_val, y_val
    Indices (lists): X_train_init_inds - inds of first training set (iteration 1)
                     X_train_current_inds - inds of labeled dataset (iteration i)
                     X_train_remaining_inds - inds of unlabeled dataset (iteration i)
                     X_train_original_inds - inds of (full) original training set
    """
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    ##############################################################
    # Load data
    ##############################################################
    X_test_ood = None
    X_train_original, y_train_original = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                          evaluate=False)
    if args.task_name == 'imdb' and os.path.exists(IMDB_CONTR_DATA_DIR):
        X_val_contrast, y_val_contrast = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                          evaluate=True, contrast=True)
        X_test_contrast, y_test_contrast = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                            test=True, contrast=True)
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)
    if args.task_name == 'mnli':
        X_test_ood, y_test_ood = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True,
                                                  ood=True)
    if args.task_name == 'imdb':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'SST-2'), 'sst-2', args.model_type,
                                                  test=True)
    if args.task_name == 'sst-2':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'IMDB'), 'imdb', args.model_type,
                                                  test=True)
    if args.task_name == 'qqp':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'MRPC'), 'mrpc', args.model_type,
                                                  test=True)
    if args.task_name == 'qnli':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'RTE'), 'rte', args.model_type,
                                                  test=True)

    X_train_original_inds = list(np.arange(len(X_train_original)))  # original pool
    X_val_inds = list(np.arange(len(X_val)))
    X_test_inds = list(np.arange(len(X_test)))

    args.binary = True if len(set(np.array(y_train_original)[X_train_original_inds])) == 2 else False
    args.num_classes = len(set(np.array(y_train_original)[X_train_original_inds]))

    if args.acquisition_size is None:
        args.acquisition_size = round(len(X_train_original_inds) / 100)  # 1%
        if args.dataset_name in ['qnli', 'ag_news']:
            args.acquisition_size = round(args.acquisition_size / 2)  # 0.5%
        elif args.dataset_name in ['dbpedia']:
            args.acquisition_size = round(len(X_train_original_inds) / 1000)  # 0.1%
    if args.init_train_data is None:
        args.init_train_data = round(len(X_train_original_inds) / 100)  # 1%
        if args.dataset_name in ['qnli', 'ag_news']:
            args.init_train_data = round(args.init_train_data / 2)  # 0.5%
        elif args.dataset_name in ['dbpedia']:
            args.init_train_data = round(len(X_train_original_inds) / 1000)  # 0.1%

    if args.indicator == "small_config":
        args.acquisition_size = 100
        args.init_train_data = 100
        args.budget = 1100

    if args.indicator == "25_config":
        args.acquisition_size = round(len(X_train_original_inds) * 2 / 100)  # 2%
        args.init_train_data = round(len(X_train_original_inds) * 1 / 100)  # 1%
        args.budget = round(len(X_train_original_inds) * 27 / 100)  # 25%

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     cache_dir=args.cache_dir,
    #     use_fast=args.use_fast_tokenizer,
    # )
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    ##############################################################
    # Stats
    ##############################################################
    print("\nDataset for annotation: {}\nAcquisition function: {}\n"
          "Budget: {}% of labeled data\n".format(args.dataset_name,
                                                 args.acquisition,
                                                 args.budget))

    # Mean and std of length of selected sequences
    if args.dataset_name in ['sst-2', 'ag_news', 'dbpedia', 'trec-6', 'imdb']:
        l = [len(x.split()) for x in np.array(X_train_original)[X_train_original_inds]]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in
             np.array(X_train_original)[X_train_original_inds]]
    else:
        NotImplementedError
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    print('Average length in words: {}'.format(length_mean))

    init_train_data = args.init_train_data
    init_train_percent = init_train_data / len(list(np.array(X_train_original)[X_train_original_inds])) * 100

    ##############################################################
    # Experiment dir
    ##############################################################
    results_per_iteration = {}

    unc_method = args.unc if args.mc_samples is None else 'mc{}'.format(args.mc_samples)
    exp_name = 'al_{}_{}_{}_{}_{}'.format(args.dataset_name,
                                          args.model_type,
                                          unc_method,
                                          args.acquisition,
                                          args.seed)
    if args.indicator is not None: exp_name += '_{}'.format(args.indicator)
    if args.bayes_output: exp_name += '_bayes'
    create_dir(AL_RES_DIR)
    results_per_iteration_dir = os.path.join(AL_RES_DIR, exp_name)
    create_dir(results_per_iteration_dir)
    resume_dir = results_per_iteration_dir

    ##############################################################
    # Resume
    ##############################################################
    if args.resume:
        if not os.path.exists(results_per_iteration_dir) or not os.listdir(results_per_iteration_dir):
            args.resume = False
            print('Experiment does not exist. Cannot resume. Start from the beginning.')

    if args.resume:
        print("Resume AL loop.....")
        with open(os.path.join(resume_dir, 'results_of_iteration.json'), 'r') as f:
            results_per_iteration = json.load(f)
        with open(os.path.join(resume_dir, 'selected_ids_per_iteration.json'), 'r') as f:
            ids_per_it = json.load(f)

        current_iteration = results_per_iteration['last_iteration'] + 1

        X_train_current_inds = []
        for key in ids_per_it:
            X_train_current_inds += ids_per_it[key]

        X_train_remaining_inds = [i for i in X_train_original_inds if i not in X_train_current_inds]
        assert len(X_train_current_inds) + len(X_train_remaining_inds) == len(
            X_train_original_inds), "current {}, remaining {}, " \
                                    "original {}".format(len(X_train_current_inds),
                                                         len(X_train_remaining_inds),
                                                         len(X_train_original_inds))

        print("Current labeled dataset {}".format(len(X_train_current_inds)))
        print("Unlabeled dataset (Dpool) {}".format(len(X_train_remaining_inds)))

        current_annotations = results_per_iteration['current_annotations']
        annotations_per_iteration = results_per_iteration['annotations_per_iteration']
        total_annotations = round(args.budget * len(X_train_original) / 100)
        if args.budget > 100: total_annotations = args.budget
        assert current_annotations <= total_annotations, "Experiment done already!"
        total_iterations = round(total_annotations / annotations_per_iteration)

        if annotations_per_iteration != args.acquisition_size:
            annotations_per_iteration = args.acquisition_size
            print("New budget! {} more iterations.....".format(
                total_iterations - round(current_annotations / annotations_per_iteration)))

        X_discarded_inds = [x for x in X_train_original_inds if x not in X_train_remaining_inds
                            and x not in X_train_current_inds]

        assert len(X_train_current_inds) + len(X_train_remaining_inds) + len(X_discarded_inds) == \
               len(X_train_original_inds), "current {}, remaining {}, discarded {}, original {}".format(
            len(X_train_current_inds),
            len(X_train_remaining_inds),
            len(X_discarded_inds),
            len(X_train_original_inds))
        assert bool(not set(X_train_current_inds) & set(X_train_remaining_inds))

        it2per = {}  # iterations to data percentage
        val_acc_previous = None
        args.acc_best_iteration = 0
        args.acc_best = 0

        print("current iteration {}".format(current_iteration))
        print("annotations_per_iteration {}".format(annotations_per_iteration))
        print("budget {}".format(args.budget))
    else:
        ##############################################################
        # New experiment!
        ##############################################################
        ##############################################################
        # Denote labeled and unlabeled datasets
        ##############################################################
        # Pool of unlabeled data: dict containing all ids corresponding to X_train_original.
        # For each id we save (1) its true labels, (2) in which AL iteration it was selected for annotation,
        # (3) its predictive uncertainty for all iterations
        # (only for the selected ids so that we won't evaluate in the entire Dpool in every iteration)
        # d_pool = {}

        # ids_per_iteration dict: contains the indices selected at each AL iteration
        ids_per_it = {}

        ##############################################################
        # Select validation data
        ##############################################################
        # for now we use the original dev set
        # al_init_prints(len(np.array(X_train_original)[X_train_original_inds]), len(np.array(X_val)[X_val_inds]),
        #                args.budget, init_train_percent)

        ##############################################################
        # Select first training data
        ##############################################################
        y_strat = np.array(y_train_original)[X_train_original_inds]

        X_train_original_after_sampling_inds = []
        X_train_original_after_sampling = []

        if args.init == 'random':
            X_train_init_inds, X_train_remaining_inds, _, _ = train_test_split(X_train_original_inds,
                                                                               np.array(y_train_original)[
                                                                                   X_train_original_inds],
                                                                               train_size=args.init_train_data,
                                                                               random_state=args.seed,
                                                                               stratify=y_strat)

        else:
            print(args.init)
            raise NotImplementedError

        ####################################################################
        # Create Dpool and Dlabels
        ####################################################################
        X_train_init = list(np.asarray(X_train_original, dtype='object')[X_train_init_inds])
        y_train_init = list(np.asarray(y_train_original, dtype='object')[X_train_init_inds])

        for i in list(set(y_train_init)):
            init_train_dist_class = 100 * np.sum(np.array(y_train_init) == i) / len(y_train_init)
            print('init % class {}: {}'.format(i, init_train_dist_class))

        if X_train_original_after_sampling_inds == []:
            assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(
                X_train_original_inds), 'init {}, remaining {}, original {}'.format(len(X_train_init_inds),
                                                                                    len(X_train_remaining_inds),
                                                                                    len(X_train_original_inds))
        else:
            assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(X_train_original_after_sampling_inds)

        ids_per_it.update({str(0): list(map(int, X_train_init_inds))})
        assert len(ids_per_it[str(0)]) == args.init_train_data

        ####################################################################
        # Annotations & budget
        ####################################################################
        current_annotations = len(X_train_init)  # without validation data
        if X_train_original_after_sampling == []:
            total_annotations = round(args.budget * len(X_train_original) / 100)
        else:
            total_annotations = round(args.budget * len(X_train_original_after_sampling) / 100)
        if args.budget > 100: total_annotations = args.budget
        annotations_per_iteration = args.acquisition_size
        total_iterations = ceil(total_annotations / annotations_per_iteration)

        X_train_current_inds = X_train_init_inds.copy()

        X_discarded_inds = [x for x in X_train_original_inds if x not in X_train_remaining_inds
                            and x not in X_train_current_inds]

        it2per = {}  # iterations to data percentage
        val_acc_previous = None
        args.acc_best_iteration = 0
        args.acc_best = 0
        current_iteration = 1

    # Assertions
    assert bool(not set(X_train_remaining_inds) & set(X_train_current_inds))

    """
        Indices of X_train_original: X_train_init_inds - inds of first training set (iteration 1)
                                     X_train_current_inds - inds of labeled dataset (iteration i)
                                     X_train_remaining_inds - inds of unlabeled dataset (iteration i)
                                     X_train_original_inds - inds of (full) original training set
                                     X_disgarded_inds - inds from dpool that are disgarded

    """

    ##############################################################
    # Active Learning loop
    ##############################################################
    while current_iteration < total_iterations + 1:

        it2per[str(current_iteration)] = round(len(X_train_current_inds) / len(X_train_original_inds), 2) * 100

        ##############################################################
        # Train model on training dataset (Dtrain)
        ##############################################################
        train_results = train_transformer_model(args=args,
                                                X_inds=X_train_current_inds,
                                                X_val_inds=X_val_inds,
                                                iteration=current_iteration,
                                                val_acc_previous=val_acc_previous)

        val_acc_previous = train_results['acc']
        print("\nDone Training!\n")

        ##############################################################
        # Test model on test data (D_test)
        ##############################################################
        print("\nStart Testing on test set!\n")
        test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True)
        test_results, test_logits = my_evaluate(test_dataset, args, train_results['model'], prefix="", mc_samples=None)
        test_results.pop('gold_labels', None)

        ##############################################################
        # Test model on OOD test data (D_ood)
        ##############################################################
        print("\nEvaluating robustness! Start testing on OOD test set!\n")
        if X_test_ood is not None and args.indicator == '25_config':
            if args.dataset_name == 'sst-2':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'imdb', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'IMDB'))
            elif args.dataset_name == 'imdb':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'sst-2', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'SST-2'))
            elif args.dataset_name == 'qqp':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'mrpc', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'MRPC'))
            elif args.dataset_name == 'qnli':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'rte', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'RTE'))
            else:
                ood_test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True, ood=True)
            ood_test_results, ood_test_logits = my_evaluate(ood_test_dataset, args, train_results['model'], prefix="",
                                                            mc_samples=None)
            ood_test_results.pop('gold_labels', None)

        ##############################################################
        # Test model on contrast + original test data (D_test_contrast)
        ##############################################################
        if args.dataset_name == 'imdb' and os.path.exists(IMDB_CONTR_DATA_DIR):
            contrast_results = contrast_acc_imdb(args, tokenizer, train_results, results_per_iteration_dir,
                                                 iteration=current_iteration)
        else:
            contrast_results = None

        ##############################################################
        # Test model on unlabeled data (Dpool)
        ##############################################################
        start = time.time()
        dpool_loss, logits_dpool, results_dpool = [], [], []
        if args.acquisition not in ['random', 'alps', 'badge', 'FTbertKM']:
            dpool_loss, logits_dpool, results_dpool = test_transformer_model(args, X_train_remaining_inds,
                                                                             model=train_results['model'])
            results_dpool.pop('gold_labels', None)
            if args.unc=="temp":
                eval_dataset = get_glue_tensor_dataset(X_val_inds, args, args.task_name, tokenizer, evaluate=True)
                temp_model = tune_temperature(eval_dataset, args, train_results['model'], return_model_temp=True)
                new_logits = temp_model.temperature_scale(logits_dpool)
                logits_dpool = new_logits
        end = time.time()
        inference_time = end - start

        ########################################################################################################
        # compute inference on the other selected input samples until this iteration (part of training set)
        ########################################################################################################
        X_rest_inds = []
        for i in range(0, current_iteration):
            X_rest_inds += ids_per_it[str(i)]
        # Assert no common data in Dlab and Dpool
        assert bool(not set(X_train_remaining_inds) & set(X_rest_inds))

        ##############################################################
        # Select unlabeled samples for annotation
        # -> annotate
        # -> update training dataset & unlabeled dataset
        ##############################################################
        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        start = time.time()
        sampled_ind, stats = calculate_uncertainty(args=args,
                                                   method=args.acquisition,
                                                   logits=logits_dpool,
                                                   annotations_per_it=annotations_per_iteration,
                                                   task=args.task_name,
                                                   candidate_inds=X_train_remaining_inds,
                                                   labeled_inds=X_train_current_inds,
                                                   discarded_inds=X_discarded_inds,
                                                   original_inds=X_train_original_inds,
                                                   X_original=X_train_original,
                                                   y_original=y_train_original)
        end = time.time()
        selection_time = end - start

        # Update results dict
        results_per_iteration[str(current_iteration)] = {'data_percent': it2per[str(current_iteration)],
                                                         'total_train_samples': len(X_train_current_inds),
                                                         'inference_time': inference_time,
                                                         'selection_time': selection_time}
        results_per_iteration[str(current_iteration)]['val_results'] = train_results
        results_per_iteration[str(current_iteration)]['test_results'] = test_results

        if X_test_ood is not None:
            results_per_iteration[str(current_iteration)]['ood_test_results'] = ood_test_results
            results_per_iteration[str(current_iteration)]['ood_test_results'].pop('model', None)

        if contrast_results is not None:
            results_per_iteration[str(current_iteration)]['contrast_test_results'] = contrast_results

        results_per_iteration[str(current_iteration)]['val_results'].pop('model', None)
        results_per_iteration[str(current_iteration)]['test_results'].pop('model', None)
        results_per_iteration[str(current_iteration)].update(stats)

        current_annotations += annotations_per_iteration

        # X_train_current_inds and X_train_remaining_inds are lists of indices of the original dataset
        # sampled_inds is a list of indices OF THE X_train_remaining_inds(!!!!) LIST THAT SHOULD BE REMOVED
        # INCEPTION %&#!@***CAUTION***%&#!@
        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
            X_train_current_inds += list(sampled_ind)
        else:
            X_train_current_inds += list(np.array(X_train_remaining_inds)[sampled_ind])

        assert len(ids_per_it[str(0)]) == args.init_train_data

        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
            selected_dataset_ids = sampled_ind
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data
        else:
            selected_dataset_ids = list(np.array(X_train_remaining_inds)[sampled_ind])
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data

        ids_per_it.update({str(current_iteration): selected_dataset_ids})

        assert len(ids_per_it[str(0)]) == args.init_train_data
        assert len(ids_per_it[str(current_iteration)]) == annotations_per_iteration

        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
            X_train_remaining_inds = [x for x in X_train_original_inds if x not in X_train_current_inds
                                      and x not in X_discarded_inds]
        else:
            X_train_remaining_inds = list(np.delete(X_train_remaining_inds, sampled_ind))

        # Assert no common data in Dlab and Dpool
        assert bool(not set(X_train_current_inds) & set(X_train_remaining_inds))

        # Assert unique (no duplicate) inds in Dlab & Dpool
        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        # Assert each list of inds unique
        set(X_train_original_inds).difference(set(X_train_current_inds))
        if args.indicator is None and args.indicator != "small_config":
            assert set(X_train_original_inds).difference(set(X_train_current_inds)) == set(
                X_train_remaining_inds + X_discarded_inds)

        results_per_iteration['last_iteration'] = current_iteration
        results_per_iteration['current_annotations'] = current_annotations
        results_per_iteration['annotations_per_iteration'] = annotations_per_iteration
        results_per_iteration['X_val_inds'] = list(map(int, X_val_inds))
        print("\n")
        print("*" * 12)
        print("End of iteration {}:".format(current_iteration))
        print("Train loss {}, Val loss {}, Test loss {}".format(train_results['train_loss'], train_results['loss'],
                                                                test_results['loss']))
        print("Annotated {} samples".format(annotations_per_iteration))
        print("Current labeled (training) data: {} samples".format(len(X_train_current_inds)))
        print("Remaining budget: {} (in samples)".format(total_annotations - current_annotations))
        print("*" * 12)
        print()

        current_iteration += 1

        print('Saving json with the results....')

        with open(os.path.join(results_per_iteration_dir, 'results_of_iteration.json'), 'w') as f:
            json.dump(results_per_iteration, f)
        with open(os.path.join(results_per_iteration_dir, 'selected_ids_per_iteration.json'), 'w') as f:
            json.dump(ids_per_it, f)

        # Check budget
        if total_annotations - current_annotations < annotations_per_iteration:
            annotations_per_iteration = total_annotations - current_annotations

        if annotations_per_iteration == 0:
            break
    print('The end!....')

    return


if __name__ == '__main__':
    import argparse
    import random

    ##########################################################################
    # Setup args
    ##########################################################################
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name", )
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
    parser.add_argument("--bayes_output", required=False, type=bool, default=False,
                        help=" if True add Bayesian classification layer (UA)")
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
    # Training args
    ##########################################################################
    parser.add_argument("--do_train", default=True, type=bool, help="If true do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_thr", default=None, type=int, help="apply min threshold to warmup steps")
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
    parser.add_argument("-patience", "--patience", required=False, type=int, default=None,
                        help="patience for early stopping (steps)")
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument("--dataset_name", default=None, required=True, type=str,
                        help="Dataset [mrpc, ag_news, qnli, sst-2]")
    parser.add_argument("--task_name", default=None, type=str, help="Task [MRPC, AG_NEWS, QNLI, SST-2]")
    parser.add_argument("--max_seq_length", default=256, type=int, help="Max sequence length")
    parser.add_argument("--data_dir", default=None, required=False, type=str,
                        help="Datasets folder")
    ##########################################################################
    # AL args
    ##########################################################################
    parser.add_argument("--acquisition", required=True,
                        type=str,
                        help="acquisition function [batch_bald, bald, least_conf, entropy, random]")
    parser.add_argument("--budget", required=False,
                        default=50, type=int,
                        help="budget \in [1,100] percent. if > 100 then it represents the total annotations")
    parser.add_argument("--mc_samples", required=False, default=None, type=int,
                        help="number of MC forward passes in calculating uncertainty estimates")
    parser.add_argument("--unc", required=False, default='vanilla', type=str,
                        help="uncertainty estimation method ['vanilla', 'mc', 'temp_scale']")
    parser.add_argument("--resume", required=False,
                        default=False,
                        type=bool,
                        help="if True resume experiment")
    parser.add_argument("--acquisition_size", required=False,
                        default=None,
                        type=int,
                        help="acquisition size at each AL iteration; if None we sample 1%")
    parser.add_argument("--init_train_data", required=False,
                        default=None,
                        type=int,
                        help="initial training data for AL; if None we sample 1%")
    parser.add_argument("--indicator", required=False,
                        # default='strat_70_30',
                        default=None,
                        type=str,
                        help="experiment indicator from []")
    parser.add_argument("--init", required=False,
                        default="random",
                        type=str,
                        help="random or alps")
    ##########################################################################
    # Server args
    ##########################################################################
    parser.add_argument("-g", "--gpu", required=False, default='0', help="gpu on which this experiment runs")
    parser.add_argument("-server", "--server", required=False, default='ford',
                        help="server on which this experiment runs")
    parser.add_argument("--debug", required=False, default=False, help="debug mode")

    args = parser.parse_args()

    # Setup
    if args.server is 'ford':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("\nThis experiment runs on gpu {}...\n".format(args.gpu))
        # VIS['enabled'] = True
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

    # Setup args
    if args.seed == None:
        seed = random.randint(1, 9999)
        args.seed = seed
    if args.task_name is None: args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR
    if args.data_dir is None:
        args.data_dir = os.path.join(DATA_DIR, args.task_name)

    args.overwrite_cache = bool(True)
    args.evaluate_during_training = True

    # Output dir
    args.output_dir = os.path.join(CKPT_DIR, '{}_{}'.format(args.dataset_name, args.model_type))
    if args.acquisition is not None:
        if args.mc_samples is not None and args.unc == 'mc':
            args.output_dir = os.path.join(args.output_dir,
                                           "mc{}-{}-{}".format(args.mc_samples, args.acquisition, args.seed))
        else:
            args.output_dir = os.path.join(args.output_dir,
                                           "{}-{}-{}".format(args.unc, args.acquisition, args.seed))
    if args.indicator is not None: args.output_dir += '-{}'.format(args.indicator)
    if args.bayes_output is not None: args.output_dir += '-bayes'
    print('output_dir={}'.format(args.output_dir))
    create_dir(args.output_dir)

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

    # Setup logging
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

    args.task_name = args.task_name.lower()

    loop(args)
