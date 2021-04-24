import glob
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, ConcatDataset
from tqdm import tqdm
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    # BertConfig,
    # BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    # RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    # XLNetForSequenceClassification,
    XLNetTokenizer,
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.general import acc, f1_macro, precision_macro, recall_macro, create_dir
from src.glue.run_glue import compute_metrics, train
from src.metrics import uncertainty_metrics
from src.temperature_scaling import tune_temperature

from src.transformers.configuration_bert import BertConfig
from src.transformers.modeling_bert import BertForSequenceClassification
from src.transformers.processors import processors, output_modes, convert_examples_to_features

from sys_config import CACHE_DIR, DATA_DIR, CKPT_DIR, RES_DIR

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (
#         BertConfig,
#         XLNetConfig,
#         XLMConfig,
#         RobertaConfig,
#         DistilBertConfig,
#         AlbertConfig,
#         XLMRobertaConfig,
#         FlaubertConfig,
#     )
#     ),
#     (),
# )

# todo add more models
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    # "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    # "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def get_glue_dataset(args, data_dir, task, model_type, evaluate=False, test=False, contrast=False, ood=False):
    """
    Loads original glue dataset.
    :param data_dir: path ../data/[task]
    :param task: glue task  ("cola", "mnli", "mnli-mm", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli",
                             "ag_news", "dbpedia", "trec-6")
    :param model_type: the type of the model we use (e.g. 'bert')
    :param train: if True return dev set
    :param evaluate: if True return dev set
    :param test: if True return test set
    :param test: if True return contrast set
    :return:
    """
    create_dir(data_dir)
    processor = processors[task]()
    output_mode = output_modes[task]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )
    # Dataset
    # Load data features from cache or dataset file
    if test:
        filename = "cached_{}_{}_original".format("test_contrast", str(task)) if contrast else "cached_{}_{}_original".format("test", str(task))
        if ood: filename += '_ood'
        cached_dataset = os.path.join(
            data_dir,
            filename
        )
    else:
        if evaluate and contrast:
            filename = "cached_{}_{}_original".format("dev_contrast", str(task))
        else:
            filename = "cached_{}_{}_original".format("dev" if evaluate else "train", str(task))
        cached_dataset = os.path.join(
            data_dir,
            filename,
        )
    if os.path.exists(cached_dataset):
        logger.info("Loading dataset from cached file %s", cached_dataset)
        dataset = torch.load(cached_dataset)
    else:
        logger.info("Creating dataset from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if test:
            if ood:
                examples = (
                    processor.get_test_examples_ood(data_dir)
                )
            else:
                examples = (
                    processor.get_contrast_examples("test") if contrast else processor.get_test_examples(data_dir)
                )
        else:
            if evaluate and contrast:
                examples = (
                    processor.get_contrast_examples("dev")
                )
            else:
                examples = (
                    processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
                )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if task in ['sst-2', 'cola', 'ag_news', 'dbpedia', 'trec-6', 'imdb']:
            X = [x.text_a for x in examples]
        elif task in ['mrpc', 'mnli', 'qnli', 'rte', 'qqp', 'wnli']:
            X = list(zip([x.text_a for x in examples], [x.text_b for x in examples]))
        # elif task == 'mnli':
        #     X = list(zip([x.text_a for x in examples], [x.text_b for x in examples]))
        else:
            print(task)
            NotImplementedError
        y = [x.label for x in examples]
        dataset = [X, y]

        logger.info("Saving dataset into cached file %s", cached_dataset)
        torch.save(dataset, cached_dataset)

        # Save Tensor Dataset
        if test:
            filename = "test_contrast" if contrast else "test"
            if ood: filename += '_ood'
            features_dataset = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}_original".format(
                    # "test",
                    filename,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
        else:
            filename = "dev" if evaluate else "train"
            if contrast: filename += "_contrast"
            features_dataset = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}_original".format(
                    # "dev" if evaluate else "train",
                    filename,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
        torch.save(features, features_dataset)

    return dataset


def get_glue_tensor_dataset(X_inds, args, task, tokenizer, train=False,
                            evaluate=False, test=False, augm=False, X_orig=None, X_augm=None, y_augm=None,
                            augm_features=None, dpool=False,
                            contrast=False, contrast_ori=False,
                            ood=False, data_dir=None):
    """
    Load tensor dataset (not original/raw).
    :param X_inds: list of indices to keep in the dataset (if None keep all)
    :param args: args
    :param task: task name
    :param tokenizer: tokenizer
    :param train: if True train dataset
    :param evaluate: if True dev dataset
    :param test: if True test dataset
    :param augm: if True augmented dataset
    :param X: augmented text (inputs)
    :param y: augmented labels (original if augmentation of labeled data) if unlabeled ?
    :return:
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if data_dir is None: data_dir = args.data_dir
    processor = processors[task.lower()]()
    output_mode = output_modes[task.lower()]
    # Load data features from cache or dataset file

    if test:
        prefix = "test"
    elif evaluate:
        prefix="dev"
    elif train:
        prefix="train"
    elif augm:
        prefix="augm"
    else:
        prefix="???"
    if contrast: prefix += "_contrast"
    if contrast_ori: prefix += "_contrast_ori"
    if ood: prefix += "_ood"

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}_original".format(
            prefix,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file): #and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if X_inds is not None:
            logger.info("Selecting subsample...")
            features = list(np.array(features)[X_inds])
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if test:
            if ood:
                examples = (processor.get_test_examples(data_dir, ood))
            else:
                examples = (processor.get_contrast_examples("test", contrast_ori) if (contrast or contrast_ori) else processor.get_test_examples(data_dir))
        elif evaluate:
            examples = (processor.get_contrast_examples("dev") if contrast else processor.get_dev_examples(args.data_dir))
        elif train:
            examples = (processor.get_train_examples(args.data_dir))
        elif augm:
            if dpool:
                augm_examples = (processor.get_augm_examples(X_augm, y_augm))

                features = convert_examples_to_features(
                    augm_examples,
                    tokenizer,
                    label_list=label_list,
                    max_length=args.max_seq_length,
                    output_mode=output_mode,
                    pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                )

                # Convert to Tensors and build dataset
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
                all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
                if output_mode == "classification":
                    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
                elif output_mode == "regression":
                    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

                dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

                return dataset
            else:
                if X_orig is None:  # DA for supervised learning
                    examples = (processor.get_augm_examples(X_augm, y_augm))
                else:  # DA for semi-supervised learning (consistency loss)
                    orig_examples = (processor.get_train_examples(args.data_dir))
                    orig_examples = [np.array(orig_examples)[i] for i in X_inds]
                    augm_examples = (processor.get_augm_examples(X_augm, y_augm))

                    assert len(orig_examples)==len(augm_examples), "orig len {}, augm len {}".format(len(orig_examples),
                                                                                                         len(augm_examples))
                    all_input_ids = []
                    all_attention_mask = []
                    all_token_type_ids = []
                    all_labels = []

                    for examples in [orig_examples, augm_examples]:
                        features = convert_examples_to_features(
                            examples,
                            tokenizer,
                            label_list=label_list,
                            max_length=args.max_seq_length,
                            output_mode=output_mode,
                            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                        )

                        all_input_ids.append(torch.tensor([f.input_ids for f in features], dtype=torch.long))
                        all_attention_mask.append(torch.tensor([f.attention_mask for f in features], dtype=torch.long))
                        all_token_type_ids.append(torch.tensor([f.token_type_ids for f in features], dtype=torch.long))
                        if output_mode == "classification":
                            all_labels.append(torch.tensor([f.label for f in features], dtype=torch.long))
                        elif output_mode == "regression":
                            all_labels.append(torch.tensor([f.label for f in features], dtype=torch.float))

                    dataset = TensorDataset(all_input_ids[0], all_attention_mask[0], all_token_type_ids[0],
                                            all_input_ids[1], all_attention_mask[1], all_token_type_ids[1],
                                            all_labels[0])

                    return dataset
        ################################################################
        if X_inds is not None:
            examples = list(np.array(examples)[X_inds])
            if hasattr(args, 'annotations_per_iteration') and hasattr(args, 'oversampling'):
                if args.oversampling and len(examples) != len(X_inds):
                    new_samples_inds = list(np.array(X_inds)[-args.annotations_per_iteration:])
                    examples += list(np.array(examples)[new_samples_inds])
                    assert len(examples) == len(X_inds)
        ################################################################
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)

    if augm_features is not None:
        # append train + augmented features (for DA supervised learning)
        features = features + augm_features

    if augm and augm_features is None:
        # return augmented features (to later append with trainset for DA supervised learning)
        return features

    else:
        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        if X_inds is not None and augm_features is None:
         assert len(dataset) == len(X_inds)
        return dataset



def my_evaluate(eval_dataset, args, model, prefix="", mc_samples=None,
                return_bert_embs=False):
    """
    Evaluate model using 'eval_dataset'.
    :param eval_dataset: tensor dataset
    :param args:
    :param model:
    :param prefix: -
    :param al_test: if True then eval_dataset is Dpool
    :param mc_samples: if not None, int with number of MC forward samples
    :return:
    """
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Sequential sampler - crucial!!!
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        bert_output_list = None

        if mc_samples is not None:
            # MC dropout
            test_losses = []
            logits_list = []
            for i in range(1, mc_samples + 1):
                test_losses_mc = []
                logits_mc = None
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    model.train()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                        if args.model_type != "distilbert":
                            inputs["token_type_ids"] = (
                                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]

                        eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs["labels"].detach().cpu().numpy()
                        logits_mc = logits
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                        logits_mc = torch.cat((logits_mc, logits), 0)
                    test_losses_mc.append(eval_loss / nb_eval_steps)
                    # logits_mc.append(logits)

                test_losses.append(test_losses_mc)
                logits_list.append(logits_mc)
                preds = None

            eval_loss = np.mean(test_losses)
            logits = logits_list
            preds = torch.mean(torch.stack(logits), 0).detach().cpu().numpy()
        else:
            # Standard inference (no MC dropout)
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    labels = inputs.pop("labels",None)
                    if return_bert_embs:
                        bert_output = model.bert(**inputs)[1]
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1


                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    # out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                    if return_bert_embs:
                        # bert_output_list = bert_output.detach().cpu().numpy()
                        bert_output_list = bert_output
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                    if return_bert_embs:
                        # bert_output_list = np.append(bert_output_list, bert_output.detach().cpu().numpy(), axis=0)
                        bert_output_list = torch.cat((bert_output_list, bert_output),0)
            eval_loss = eval_loss / nb_eval_steps
            logits = torch.tensor(preds)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)

            # accuracy = round(acc(out_label_ids, preds), 4)
            # f1 = round(f1_macro(out_label_ids, preds), 4)
            # precision = round(precision_macro(out_label_ids, preds), 4)
            # recall = round(recall_macro(out_label_ids, preds), 4)

            # calibration scores
            calibration_scores = uncertainty_metrics(logits, out_label_ids,
                                                     num_classes=args.num_classes)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            calibration_scores = {}
            accuracy, f1, precision, recall = 0., 0., 0., 0.,
            calibration_scores = None
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # results.update({'f1_macro': f1, 'recall': recall, 'precision': precision, 'loss': eval_loss})
        results.update(calibration_scores)
        results.update({'bert_output': bert_output_list})
        results.update({'gold_labels': out_label_ids.tolist()})
    results['loss'] = eval_loss
    # return results, eval_loss, accuracy, f1, precision, recall, logits
    return results, logits


def train_transformer(args, train_dataset, eval_dataset, model, tokenizer):
    """
    Train a transformer model.
    :param args: args
    :param train_dataset: train (tensor) dataset
    :param eval_dataset: dev (tensor) dataset
    :param model: model to train
    :param tokenizer: tokenizer
    :return:
    """
    # 10% warmup
    args.warmup_steps = int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs / 10)
    if hasattr(args, "warmup_thr"):
        if args.warmup_thr is not None:
            args.warmup_steps = min(int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs / 10), args.warmup_thr)


    print("warmup steps: {}".format(args.warmup_steps))
    print("total steps: {}".format(int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs)))
    print("logging steps: {}".format(args.logging_steps))

    ##############################
    # Train model
    ##############################
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    global_step, tr_loss, val_acc, val_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # AYTO GIATI TO KANW?
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        # checkpoints = [args.output_dir]
        checkpoints = [args.current_output_dir]
        # if args.eval_all_checkpoints:
        #     checkpoints = list(
        #         # os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        #         os.path.dirname(c) for c in
        #         sorted(glob.glob(args.current_output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        #     )
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result, logits = my_evaluate(eval_dataset, args, model, prefix=prefix)
            # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            # results.update(result)

    eval_loss = val_loss

    return model, tr_loss, eval_loss, result


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
    parser.add_argument("--data_dir", default=None, required=False, type=str,
                        help="Datasets folder")
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
    if args.data_dir is None:
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
        use_adapter=args.use_adapter,
        use_bayes_adapter=args.use_bayes_adapter,
        adapter_initializer_range=0.0002 if args.indicator=='identity_init' else 1,
        bayes_output=args.bayes_output,
        unfreeze_adapters=args.unfreeze_adapters

    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    #########################################
    # Check if experiment already done
    #########################################
    create_dir(RES_DIR)
    path = os.path.join(RES_DIR, '{}_{}_100%'.format(args.task_name, args.model_type))
    create_dir(path)
    name = 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(args.seed, args.learning_rate,
                                                    args.per_gpu_train_batch_size,
                                                    int(args.num_train_epochs))
    # if args.use_adapter: name += '_adapters'
    if args.indicator is not None: name += '_{}'.format(args.indicator)
    print(name)

    dirname = os.path.join(path, name)

    if os.path.isdir(dirname) and os.listdir(dirname) and not args.bayes_output:
        print('Experiment done!')
        exit()
    create_dir(dirname)

    #########################################
    # Load (raw) dataset
    #########################################
    X_train, y_train = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=False)
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)

    X_orig = X_train  # original train set
    y_orig = y_train  # original labels
    X_inds = list(np.arange(len(X_orig)))  # indices to original train set
    X_unlab_inds = []  # indices of ulabeled set to original train set

    args.binary = True if len(set(y_train)) == 2 else False
    args.num_classes = len(set(y_train))

    # The following code is in case we want to undersample the dataset to evaluate uncertainty
    # in low (data) resource scenarios
    args.undersampling = False
    if args.indicator is not None:
        # Undersample training dataset (stratified sampling)
        if "sample_" in args.indicator:
            args.undersampling = True
            num_to_sample = int(args.indicator.split("_")[1])
            X_train_orig_after_sampling_inds, X_train_orig_remaining_inds, _, _ = train_test_split(
                X_inds,
                y_orig,
                train_size=num_to_sample,
                random_state=args.seed,
                stratify=y_train)
            X_inds = X_train_orig_after_sampling_inds   # indices of train set to original train set
            # X_train = list(np.array(X_train)[X_inds])   # train set
            # y_train = list(np.array(y_train)[X_inds])   # labels
            # Treat the rest of training data as unlabeled data
            X_unlab_inds = X_train_orig_remaining_inds  # indices of ulabeled set to original train set

    assert len(X_unlab_inds) + len(X_inds) == len(X_orig)
    assert bool(not (set(X_unlab_inds) & set(X_inds)))
    assert max(X_inds) < len(X_orig)
    if X_unlab_inds != []:
        assert max(X_unlab_inds) < len(X_orig)

    #########################################
    # Load (tensor) dataset
    #########################################
    train_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
    assert len(train_dataset) == len(X_inds)
    eval_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, evaluate=True)
    test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True)
    if args.dataset_name == 'mnli':
        test_dataset_ood = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True, ood=True)

    #######################
    # Train setup
    #######################
    # select after how many steps will evaluate during training so that we will evaluate at least 5 times in one epoch
    minibatch = int(len(X_inds) / (args.per_gpu_train_batch_size * max(1, args.n_gpu)))
    args.logging_steps = min(int(minibatch / 5), 500)
    if args.logging_steps < 1:
        args.logging_steps = 1

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #######################
    # Train
    #######################
    model, tr_loss, val_loss, val_results = train_transformer(args, train_dataset, eval_dataset, model, tokenizer)

    #######################
    # Test
    #######################
    # comment out because we do it later ("vanilla")
    # test_results, test_logits = my_evaluate(test_dataset, args, model, prefix="", mc_samples=None)

    # print('Saving json with the results....')
    #
    # results = {"val_results": val_results, "test_results": test_results}
    # with open(os.path.join(dirname, 'results.json'), 'w') as f:
    #     json.dump(results, f)

    #######################
    # Test uncertainty
    #######################
    print('Evaluate uncertainty on dev & test sets....')
    if args.test_all_uncertainty:
        # Vanilla
        print('Evaluate vanilla....')
        vanilla_results_val, vanilla_val_logits = my_evaluate(eval_dataset, args, model, mc_samples=None)
        vanilla_results_test, vanilla_test_logits = my_evaluate(test_dataset, args, model, mc_samples=None)
        vanilla_results = {"val_results": vanilla_results_val, "test_results": vanilla_results_test}
        # if args.dataset_name == 'mnli':
        #     print('Evaluate OOD!....')
        #     vanilla_results_test_ood, vanilla_test_logits_ood = my_evaluate(test_dataset_ood, args, model, mc_samples=None)
        #     vanilla_results['test_results_ood'] = vanilla_results_test_ood
        filename = 'vanilla_results'
        if args.use_adapter: filename += '_adapter'
        if args.use_bayes_adapter: filename += '_bayes_adapter'
        if args.bayes_output: filename += '_bayes_output'
        if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
        with open(os.path.join(dirname, '{}.json'.format(filename)), 'w') as f:
            json.dump(vanilla_results, f)
        # Monte Carlo dropout
        for m in [3,5,10,20]:
            print('Evaluate MC dropout (N={})....'.format(m))
            mc_results_val, _ = my_evaluate(eval_dataset, args, model, mc_samples=m)
            mc_results_test, _ = my_evaluate(test_dataset, args, model, mc_samples=m)
            mc_results = {"val_results": mc_results_val, "test_results": mc_results_test}
            # if args.dataset_name == 'mnli':
            #     print('Evaluate OOD!....'.format(m))
            #     mc_results_test_ood, _ = my_evaluate(test_dataset_ood, args, model, mc_samples=m)
            #     mc_results['test_results_ood'] = mc_results_test_ood
            filename = 'mc{}_results'.format(m)
            if args.use_adapter: filename += '_adapter'
            if args.use_bayes_adapter: filename += '_bayes_adapter'
            if args.bayes_output: filename += '_bayes_output'
            if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
            with open(os.path.join(dirname, '{}.json'.format(filename)), 'w') as f:
                json.dump(mc_results, f)
        # Temperature Scaling
        print('Evaluate temperature scaling....')
        temp_model = tune_temperature(eval_dataset, args, model, return_model_temp=True)
        temp_scores_val = temp_model.temp_scale_metrics(args.task_name, vanilla_val_logits,
                                                        vanilla_results_val['gold_labels'])
        temp_scores_test = temp_model.temp_scale_metrics(args.task_name, vanilla_test_logits,
                                                        vanilla_results_test['gold_labels'])
        temp_scores = {"val_results": temp_scores_val, "test_results": temp_scores_test}
        # if args.dataset_name == 'mnli':
        #     print('Evaluate OOD!....'.format(m))
        #     temp_scores_test_ood = temp_model.temp_scale_metrics(args.task_name, vanilla_test_logits_ood,
        #                                                      vanilla_results_test_ood['gold_labels'])
        #     temp_scores['test_results_ood'] = temp_scores_test_ood
        filename = 'temp_scale_results'
        if args.use_adapter: filename += '_adapter'
        if args.use_bayes_adapter: filename += '_bayes_adapter'
        if args.bayes_output: filename += '_bayes_output'
        if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
        with open(os.path.join(dirname, '{}.json'.format(filename)), 'w') as f:
            json.dump(temp_scores, f)

    #######################
    # Test uncertainty OOD
    #######################
    if args.dataset_name == 'mnli':
        test_dataset_ood = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True, ood=True)
    elif args.dataset_name == 'qqp':
        test_dataset_ood = get_glue_tensor_dataset(None, args, 'mrpc', tokenizer, test=True,
                                                   data_dir=os.path.join(DATA_DIR, 'MRPC'))
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
        raise NotImplementedError

        #######################
        # Test uncertainty
        #######################
    print('Evaluate uncertainty on dev & test sets....')
    if args.test_all_uncertainty:
        # Vanilla
        print('Evaluate OOD....')
        vanilla_ood_results, vanilla_ood_logits = my_evaluate(test_dataset_ood, args, model, mc_samples=None)
        vanilla_results = {"test_ood_results": vanilla_ood_results}
        filename = 'vanilla_results'
        if args.use_adapter: filename += '_adapter'
        if args.use_bayes_adapter: filename += '_bayes_adapter'
        if args.bayes_output: filename += '_bayes_output'
        if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
        with open(os.path.join(dirname, '{}_ood.json'.format(filename)), 'w') as f:
            json.dump(vanilla_results, f)
        # Monte Carlo dropout
        for m in [3, 5, 10, 20]:
            print('Evaluate MC dropout (N={})....'.format(m))
            mc_ood_results, _ = my_evaluate(test_dataset_ood, args, model, mc_samples=m)
            mc_results = {"test_ood_results": mc_ood_results}
            filename = 'mc{}_results'.format(m)
            if args.use_adapter: filename += '_adapter'
            if args.use_bayes_adapter: filename += '_bayes_adapter'
            if args.bayes_output: filename += '_bayes_output'
            if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
            with open(os.path.join(dirname, '{}_ood.json'.format(filename)), 'w') as f:
                json.dump(mc_results, f)
        # Temperature Scaling
        print('Evaluate temperature scaling....')
        # with open(dirname) as json_file:
        #     results = json.load(json_file)
        #     temperature = results['val_results']['temperature']
        temperature = temp_scores_val['temperature']
        temp_model = tune_temperature(test_dataset_ood, args, model, return_model_temp=True)
        temp_ood_scores = temp_model.temp_scale_metrics(args.task_name, vanilla_ood_logits,
                                                        vanilla_ood_results['gold_labels'],
                                                        temperature=temperature)
        temp_scores = {"test_ood_results": temp_ood_scores}
        filename = 'temp_scale_results'
        if args.use_adapter: filename += '_adapter'
        if args.use_bayes_adapter: filename += '_bayes_adapter'
        if args.bayes_output: filename += '_bayes_output'
        if (args.use_adapter or args.bayes_output) and args.unfreeze_adapters: filename += '_unfreeze'
        with open(os.path.join(dirname, '{}_ood.json'.format(filename)), 'w') as f:
            json.dump(temp_scores, f)
