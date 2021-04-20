import collections
import logging
import math
import os
import sys

import numpy as np
import torch

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


def variation_ratios(logits_b_K_C):
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) - torch.exp(
        torch.max(logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0]
    )


def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


def max_entropy_acquisition_function(logits_b_K_C):
    return entropy(logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=-1)


def mutual_information(logits_B_K_C):
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def mean_stddev(logits_B_K_C):
    stddev_B_C = torch.std(torch.exp(logits_B_K_C).double(), dim=1, keepdim=True).squeeze(1)
    return torch.mean(stddev_B_C, dim=1, keepdim=True).squeeze(1)


def mean_stddev_acquisition_function(logits_b_K_C):
    return mean_stddev(logits_b_K_C)


# def select_alps(args, sampled, acquisition_size, dpool_inds=None, original_inds=None):
#     torch.cuda.empty_cache()
#     # Model
#     if args.tapt is None:
#         # original bert
#         if args.config_name:
#             config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
#         elif args.model_name_or_path:
#             config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
#         else:
#             raise ValueError
#         model = AutoModelWithLMHead.from_pretrained(
#             args.model_name_or_path,
#             from_tf=bool(".ckpt" in args.model_name_or_path),
#             config=config,
#             cache_dir=args.cache_dir,
#         )
#
#     else:
#         # tapt bert
#         model_dir = os.path.join(CKPT_DIR, '{}_ft'.format(args.dataset_name), args.tapt)
#         model = AutoModelWithLMHead.from_pretrained(model_dir)
#
#     if args.local_rank == 0:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
#
#     model.to(args.device)
#
#     # Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         cache_dir=args.cache_dir,
#         use_fast=args.use_fast_tokenizer,
#     )
#
#     args.task_name = args.task_name.lower()
#     if args.task_name not in processors:
#         raise ValueError("Task not found: %s" % (args.task_name))
#
#     # Dataset
#     # dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, train=True, evaluate=False)
#     dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True, evaluate=False)
#     # X_inds=None because they are removed later (the labeled/already sampled data)
#
#     if sampled == []:
#         sampled = torch.LongTensor([])
#     else:
#         sampled = torch.LongTensor(sampled)
#
#     args.sampling = 'alps'
#     args.query_size = acquisition_size
#     args.mlm_probability = 0.15
#     args.head = sampling_to_head(args.sampling)
#
#     logger.info(f"Already sampled {len(sampled)} examples")
#     sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)
#
#     torch.cuda.empty_cache()
#     # assert len(list(np.array(original_inds)[sampled_ids]))==acquisition_size
#     # return list(sampled_ids)
#     return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original
#
# def badge(args, sampled, acquisition_size, model, dpool_inds, original_inds=None):
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         cache_dir=args.cache_dir,
#         use_fast=args.use_fast_tokenizer,
#     )
#
#     # Dataset
#     # dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, train=True)
#     dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True, evaluate=False)
#     # X_inds=None because they are removed later (the labeled/already sampled data)
#
#     if sampled == []:
#         sampled = torch.LongTensor([])
#     else:
#         sampled = torch.LongTensor(sampled)
#
#     args.sampling = 'badge'
#     args.query_size = acquisition_size
#     args.mlm_probability = 0.15
#     args.head = sampling_to_head(args.sampling)
#
#     logger.info(f"Already sampled {len(sampled)} examples")
#     sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)
#
#     torch.cuda.empty_cache()
#
#     # assert len(list(np.array(original_inds)[sampled_ids]))==acquisition_size
#     # return list(sampled_ids)
#     return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original
#
# def bertKM(args, sampled, acquisition_size, model, dpool_inds, original_inds=None):
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         cache_dir=args.cache_dir,
#         use_fast=args.use_fast_tokenizer,
#     )
#
#     # Dataset
#     # dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, train=True)
#     dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True)
#     # X_inds=None because they are removed later (the labeled/already sampled data)
#
#     if sampled == []:
#         sampled = torch.LongTensor([])
#     else:
#         sampled = torch.LongTensor(sampled)
#
#     args.sampling = 'FTbertKM'
#     args.query_size = acquisition_size
#     args.mlm_probability = 0.15
#     args.head = sampling_to_head(args.sampling)
#
#     logger.info(f"Already sampled {len(sampled)} examples")
#     sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)
#
#     torch.cuda.empty_cache()
#
#     # assert len(list(np.array(original_inds)[sampled_ids]))==acquisition_size, 'sampled {}, acquisition size {}'.format(len(list(np.array(original_inds)[sampled_ids])),
#     #                                                                                                            acquisition_size)
#     # return list(sampled_ids)
#     return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original
#     # return list(np.array(original_inds)[sampled_ids])

def least_confidence_af(logits):
    """
    Least Confidence (LC) acquisition function.
    Calculates the difference between the most confident prediction and 100% confidence.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return: list with the confidence scores [0,1] for all samples
             with 1: most uncertain/least confident
    """

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det
    least_conf_ = variation_ratios(logits_B_K_C)
    return least_conf_.cpu().numpy()


def margin_of_confidence_af(prob_dist):
    """
    Margin of confidence acquisition function.
    Calculates the difference between the top two most confident predictions.
    (works for > 2 classes. for 2 classes it is identical to least confidence)
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    prob = torch.sort(prob_dist, descending=True)
    difference = [prob.values[x][0] - prob.values[x][1] for x
                  in range(0, len(prob_dist))]
    margin_conf = 1 - np.array(difference)
    return margin_conf


def ratio_of_confidence_af(prob_dist):
    """
    Ratio of confidence acquisition function.
    Calculates the ratio between the top two most confident predictions.
    (works for > 2 classes. for 2 classes it is identical to least confidence)
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    prob = torch.sort(prob_dist, descending=True)
    ratio_conf = np.array([prob.values[x][1] / prob.values[x][0] for x
                           in range(0, len(prob_dist))])
    return ratio_conf


def entropy_af(logits):
    """
    Entropy-based uncertainty.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    # if type(prob_dist) is list:
    #     prob_dist = torch.mean(torch.stack(prob_dist), 0)  # mean of N MC stochastic passes
    # prbslogs = prob_dist * torch.log2(prob_dist)
    # numerator = 0 - torch.sum(prbslogs, dim=1)
    # denominator = math.log2(prob_dist.size(1))
    # entropy_scores = numerator / denominator

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    entropy_scores_ = max_entropy_acquisition_function(logits_B_K_C)
    return entropy_scores_.cpu().numpy()


def bald_af(logits):
    """
    Bayesian Active Learning by Disagreement (BALD).
    paper: https://arxiv.org/abs/1112.5745
    :param prob_dist:
    :return:
    """
    # # my way
    # # entropy
    # assert type(prob_dist) == list
    # mean_MC_prob_dist = torch.mean(torch.stack(prob_dist), 0)     # mean of N MC stochastic passes
    # prbslogs = mean_MC_prob_dist * torch.log2(mean_MC_prob_dist)  # p logp
    # numerator = 0 - torch.sum(prbslogs, dim=1)                    # -sum p logp
    # denominator = math.log2(mean_MC_prob_dist.size(1))            # class normalisation
    #
    # entropy = numerator / denominator
    #
    # # expectation of entropy
    # prob_dist_tensor = torch.stack(prob_dist, dim=-1)                                  # of shape (#samples, C, N)
    # classes_sum = torch.sum(prob_dist_tensor * torch.log2(prob_dist_tensor), dim=-1)   # of shape (#samples, C)
    # MC_sum = torch.sum(classes_sum, -1)                                                # of shape (#samples)
    #
    # expectation_of_entropy = MC_sum
    #
    # mutual_information_ = entropy + expectation_of_entropy

    # bb way
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)
    bald_scores = mutual_information(logits_B_K_C)

    return bald_scores.cpu().numpy()


def mean_std_af(logits):
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    scores = mean_stddev_acquisition_function(logits_B_K_C)
    return scores.cpu().numpy()


def calculate_uncertainty(args, method, logits, annotations_per_it, task=None,
                          candidate_inds=None,
                          labeled_inds=None,
                          discarded_inds=None,
                          original_inds=None,
                          X_original=None, y_original=None):
    """
    Selects and performs uncertainty-based acquisition.
    :param method: uncertainty-based acquisition function. options:
        - 'least_conf' for least confidence
        - 'margin_conf' for margin of confidence
        - 'ratio_conf' for ratio of confidence
    :param prob_dist: output probability distribution
    :param logits: output logits
    :param annotations_per_it: number of samples (to be sampled)
    :param D_lab: [(X_labeled, y_labeled)] labeled data
    :param D_unlab: [(X_unlabeled, y_unlabeled)] unlabeled data
    :return:
    """
    #
    prob_dist = None

    init_labeled_data = len(labeled_inds)  # before selecting data for annotation
    init_unlabeled_data = len(candidate_inds)

    if method not in ['random', 'alps', 'badge', 'FTbertKM']:
        if type(logits) is list and logits != []:
            assert init_unlabeled_data == logits[0].size(0), "logits {}, inital unlabaled data {}".format(
                logits[0].size(0), init_unlabeled_data)
        elif type(logits) != []:
            assert init_unlabeled_data == len(logits)

    if method == 'least_conf':
        uncertainty_scores = least_confidence_af(logits)
    elif method == 'ratio_conf':
        uncertainty_scores = ratio_of_confidence_af(prob_dist)
    elif method == 'margin_conf':
        uncertainty_scores = margin_of_confidence_af(prob_dist)
    elif method == 'entropy':
        uncertainty_scores = entropy_af(logits)
    elif method == 'std':
        uncertainty_scores = mean_std_af(logits)
    elif method == 'bald':
        uncertainty_scores = bald_af(logits)
    # elif method == 'batch_bald':
    #     uncertainty_scores, sampled_ind = BB_acquisition(logits, device, annotations_per_it)
    #     assert len(set(sampled_ind)) == len(sampled_ind), "unique {}, total {}".format(len(set(sampled_ind)),
    #                                                                                    len(sampled_ind))
    #     assert len(sampled_ind) == annotations_per_it, "sampled ind {}, acquisition size {}".format(len(sampled_ind),
    #                                                                                                 annotations_per_it)
    elif method == 'random':
        pass
    elif method == 'alps':
        pass
    elif method == 'badge':
        pass
    elif method == 'FTbertKM':
        pass
    else:
        raise ValueError('Acquisition function {} not implemented yet check again!'.format(method))

    if method == 'random':
        sampled_ind = np.random.choice(init_unlabeled_data, annotations_per_it, replace=False)
    # elif method == 'alps':
    #     old_labeled = labeled_inds.copy()
    #     old_and_new = select_alps(args, labeled_inds, annotations_per_it, candidate_inds, original_inds=original_inds)
    #     # old_and_new = select_alps(args, labeled_inds, annotations_per_it, candidate_inds)
    #     assert len(old_and_new) == len(labeled_inds) + annotations_per_it, 'old_and_new {}, labeled {}, annotations per it {}'.format(len(old_and_new), len(labeled_inds), annotations_per_it)
    #     sampled_ind = [x for x in old_and_new if x not in old_labeled]
    #     assert len(sampled_ind) == annotations_per_it, 'sampled {}, annotations per it {}'.format(len(sampled_ind), len(annotations_per_it))
    # elif method == 'badge':
    #     old_labeled = labeled_inds.copy()
    #     old_and_new = badge(args, labeled_inds, annotations_per_it, model, candidate_inds, original_inds=original_inds)
    #     # old_and_new = badge(args, labeled_inds, annotations_per_it, model, candidate_inds)
    #     assert len(old_and_new) == len(labeled_inds) + annotations_per_it, 'old_and_new {}, labeled {}, annotations per it {}'.format(len(old_and_new), len(labeled_inds), annotations_per_it)
    #     sampled_ind = [x for x in old_and_new if x not in old_labeled]
    #     assert len(sampled_ind) == annotations_per_it, 'sampled {}, annotations per it {}'.format(len(sampled_ind), len(annotations_per_it))
    # elif method == 'FTbertKM':
    #     old_labeled = labeled_inds.copy()
    #     old_and_new = bertKM(args, labeled_inds, annotations_per_it, model, candidate_inds, original_inds=original_inds)
    #     assert len(old_and_new) == len(labeled_inds) + annotations_per_it, 'old_and_new {}, labeled {}, annotations per it {}'.format(len(old_and_new), len(labeled_inds), annotations_per_it)
    #     sampled_ind = [x for x in old_and_new if x not in old_labeled]
    #     assert len(sampled_ind) == annotations_per_it, 'sampled {}, annotations per it {}'.format(len(sampled_ind), len(annotations_per_it))
    else:
        if method != 'batch_bald':
            # find indices with #samples_to_annotate least confident samples = BIGGER numbers in uncertainty_scores list
            sampled_ind = np.argpartition(uncertainty_scores, -annotations_per_it)[-annotations_per_it:]

    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[candidate_inds]
    y_unlab = np.asarray(y_original, dtype='object')[candidate_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    if method not in ['alps', 'badge', 'FTbertKM']:
        new_samples = np.asarray(X_unlab, dtype='object')[sampled_ind]
        new_labels = np.asarray(y_unlab, dtype='object')[sampled_ind]
    else:
        new_samples = np.asarray(X_original, dtype='object')[sampled_ind]
        new_labels = np.asarray(y_original, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    if task in ['sst-2', 'ag_news', 'dbpedia', 'trec-6', 'imdb']:
        l = [len(x.split()) for x in new_samples]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in new_samples]
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    if method not in ['alps', 'badge', 'FTbertKM']:
        labels_list_selected = list(np.array(y_unlab)[sampled_ind])
    else:
        labels_list_selected = list(np.array(y_original)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(new_samples) == annotations_per_it, 'len(new_samples)={}, annotatations_per_it={}'.format(
        len(new_samples), annotations_per_it)
    if args.indicator is not None:
        pass
    else:
        # assert len(labeled_inds) + len(candidate_inds) + len(discarded_inds) == len(X_original)
        assert len(labeled_inds) + len(candidate_inds) + len(discarded_inds) == len(original_inds)

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous}

    return sampled_ind, stats

# if __name__ == '__main__':
#     # prob_dist = [torch.rand(100, 2), torch.rand(100, 2), torch.rand(100, 2)]
#     MC_1 = torch.tensor(np.array([0.5, 0.5]))
#     MC_2 = torch.tensor(np.array([0.6, 0.4]))
#     MC_3 = torch.tensor(np.array([0.3, 0.7]))
#
#     prob_dist = [MC_1, MC_2, MC_3]
#     # test entropy function:
#     scores = bald(prob_dist)
#     print()
