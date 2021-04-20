import os
import pickle
import sys

import numpy as np

from main_transformer import get_glue_tensor_dataset, my_evaluate

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def contrast_acc_imdb(args, tokenizer, train_results, results_per_iteration_dir, iteration):
    print("\nStart Testing on contrast test set!\n")
    test_contrast_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True, contrast=True)
    test_contrast_results, test_contrast_logits = my_evaluate(test_contrast_dataset, args, train_results['model'],
                                                              prefix="", mc_samples=None)
    test_contrast_ori_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True,
                                                        contrast_ori=True)
    test_contrast_ori_results, test_contrast_ori_logits = my_evaluate(test_contrast_ori_dataset, args,
                                                                      train_results['model'], prefix="",
                                                                      mc_samples=None)

    res_ori = list(map(lambda x, y: x == y, np.argmax(test_contrast_ori_logits, axis=1).numpy(),
                       test_contrast_ori_results['gold_labels']))
    res_contr = list(map(lambda x, y: x == y, np.argmax(test_contrast_logits, axis=1).numpy(),
                         test_contrast_results['gold_labels']))
    consistency_acc_list = list(map(lambda x, y: x == y, res_ori,
                                    res_contr))
    consistency_acc = round(sum(consistency_acc_list) / len(consistency_acc_list), 3)
    contrast_results = {'ori_preds': list(map(int, np.argmax(test_contrast_ori_logits, axis=1).numpy())),
                        'ori_gold': list(test_contrast_ori_results['gold_labels']),
                        'contr_preds': list(map(int, np.argmax(test_contrast_logits, axis=1).numpy())),
                        'contr_gold': list(test_contrast_results['gold_labels']),
                        'consistency_acc': float(consistency_acc),
                        'test_contrast_acc': float(test_contrast_results['acc']),
                        'test_ori_acc': float(test_contrast_ori_results['acc']),
                        'test_contrast_ece': test_contrast_results['ece'],
                        'test_ori_ece': test_contrast_ori_results['ece']}

    with open(os.path.join(results_per_iteration_dir, 'contrast_set_results_{}.json'.format(iteration)),
              'wb') as handle:
        pickle.dump(contrast_results,
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    return contrast_results
