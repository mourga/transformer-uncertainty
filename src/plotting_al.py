import json
import os

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from src.general import create_dir
from sys_config import AL_RES_DIR, RES_DIR, BASE_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def read_results_json(dataset, model, af, seeds=[964, 131, 821, 12, 71],
                      unc='vanilla',
                      indicator=None, indicator_df=False):
    df = pd.DataFrame()

    keys_excluded = ['X_train_current_inds', 'X_train_remaining_inds', 'last_iteration', 'current_annotations',
                     'annotations_per_iteration', 'X_val_inds', 'dpool_augm_inds']
    for seed in seeds:
        print(dataset, seed, af)
        path = os.path.join(AL_RES_DIR, 'al_{}_{}_{}_{}_{}'.format(dataset, model, unc, af, seed))
        if indicator is not None: path += '_{}'.format(indicator)
        if os.path.exists(path):
            results_file = os.path.join(path, 'results_of_iteration') + '.json'
            ids_file = os.path.join(path, 'selected_ids_per_iteration') + '.json'

            if os.path.isfile(results_file):
                with open(results_file) as json_file:
                    results = json.load(json_file)

                    if 'test_results' in results['1'].keys():
                        _iterations = [i for i in list(results.keys()) if i not in keys_excluded]

                        iterations = [x for x in _iterations if int(x) < 52]
                        data_percent = [results[i]['data_percent'] for i in iterations]
                        samples = [results[i]['total_train_samples'] for i in iterations]
                        inf_time = [results[i]['inference_time'] for i in iterations]
                        sel_time = [results[i]['selection_time'] for i in iterations]
                        test_acc = [round(results[i]['test_results']['acc'] * 100, 2) for i in iterations]

                        val_acc = [round(results[i]['val_results']['acc'] * 100, 2) for i in iterations]

                        classes_selected = [results[i]['class_selected_samples'] for i in iterations]
                        classes_before = [results[i]['class_samples_before'] for i in iterations]
                        classes_after = [results[i]['class_samples_after'] for i in iterations]

                        ece = [results[i]['test_results']['ece']['ece'] for i in iterations]
                        entropy = [results[i]['test_results']['entropy']['mean'] for i in iterations]
                        nll = [results[i]['test_results']['nll']['mean'] for i in iterations]
                        brier = [results[i]['test_results']['brier']['mean'] for i in iterations]
                        # ood_results = None
                        if 'ood_test_results' in results['1'].keys():
                            ood_results = [round(results[i]['ood_test_results']['acc'] * 100, 2) for i in iterations]
                        else:
                            ood_results = [None for i in iterations]
                        if 'contrast_test_results' in results['1'].keys():
                            consistency_test_results = [
                                round(results[i]['contrast_test_results']['consistency_acc'] * 100, 2) for i in
                                iterations]
                            contrast_test_results = [
                                round(results[i]['contrast_test_results']['test_contrast_acc'] * 100, 2) for i in
                                iterations]
                            ori_test_results = [round(results[i]['contrast_test_results']['test_ori_acc'] * 100, 2) for
                                                i in iterations]
                        else:
                            consistency_test_results = None
                            contrast_test_results = None
                            ori_test_results = None
                        if af == "adv":
                            advs = [results[i]['num_adv'] for i in iterations]
                        else:
                            advs = [None for i in iterations]
                        num_val_adv = None
                        if 'val_adv_inds' in results['1']['val_results'].keys():
                            num_val_adv = [len(results[i]['val_results']['val_adv_inds']) for i in iterations]
                        df_ = pd.DataFrame(
                            {'iterations': iterations, 'val_acc': val_acc, 'test_acc': test_acc,
                             'ori_acc': ori_test_results, 'contrast_acc': contrast_test_results,
                             'consistency_acc': consistency_test_results,
                             'ood_test_acc': ood_results, 'data_percent': data_percent,
                             'samples': samples,
                             # 'inference_time': inf_time, 'selection_time': sel_time,
                             # 'classes_after': classes_after, 'classes_before': classes_before,
                             # 'classes_selected': classes_selected,
                             'ece': ece, 'entropy': entropy, 'nll': nll, 'brier': brier,
                             # 'num_adv': advs, 'num_val_advs':num_val_adv
                             })
                        df_['seed'] = seed
                        df_['dataset'] = dataset
                        df_['acquisition'] = af
                        df_['unc'] = unc
                        df = df.append(df_, ignore_index=True)
                    else:
                        if 'ece' not in results['1'].keys():
                            break
                        _iterations = [i for i in list(results.keys()) if i not in keys_excluded]

                        iterations = [x for x in _iterations if int(x) < 52]

                        train_loss = [results[i]['train_loss'] for i in iterations]
                        val_loss = [results[i]['loss'] for i in iterations]
                        val_acc = [round(results[i]['acc'] * 100, 2) for i in iterations]
                        if 'f1_macro' in results['1']:
                            val_f1 = [round(results[i]['f1_macro'] * 100, 2) for i in iterations]
                        elif 'f1' in results['1']:
                            val_f1 = [round(results[i]['f1'] * 100, 2) for i in iterations]
                        data_percent = [results[i]['data_percent'] for i in iterations]
                        samples = [results[i]['total_train_samples'] for i in iterations]
                        inf_time = [results[i]['inference_time'] for i in iterations]
                        sel_time = [results[i]['selection_time'] for i in iterations]
                        times_trained = [results[i]['times_trained'] for i in iterations]
                        classes_selected = [results[i]['class_selected_samples'] for i in iterations]
                        classes_before = [results[i]['class_samples_before'] for i in iterations]
                        classes_after = [results[i]['class_samples_after'] for i in iterations]
                        ece = [results[i]['ece']['ece'] for i in iterations]
                        entropy = [results[i]['entropy']['mean'] for i in iterations]
                        nll = [results[i]['nll']['mean'] for i in iterations]
                        brier = [results[i]['brier']['mean'] for i in iterations]

                        df_ = pd.DataFrame(
                            {'iterations': iterations, 'val_acc': val_acc, 'val_f1': val_f1,
                             'data_percent': data_percent,
                             'samples': samples, 'inference_time': inf_time, 'selection_time': sel_time,
                             'classes_after': classes_after, 'classes_before': classes_before,
                             'classes_selected': classes_selected,
                             'ece': ece, 'entropy': entropy, 'nll': nll, 'brier': brier})
                        df_['seed'] = seed
                        df_['dataset'] = dataset
                        df_['acquisition'] = af
                        df_['training'] = 'SL'
                        if 'uda' in indicator:
                            df_['training'] = 'CT'
                        df = df.append(df_, ignore_index=True)
    if indicator_df:
        if indicator is None: indicator = 'baseline'
        df['indicator'] = indicator

    return df


def al_plot(dataset, model='bert',
            af=['entropy', 'random'],
            seeds=[],
            unc='vanilla',
            plot_dir=None,
            indicator=None,
            y='acc',
            legend=None,
            test=True,
            ood=False,
            contrast=False):
    sns.set_style("whitegrid")

    # Choose path to save figure
    dataset_dir = os.path.join(AL_RES_DIR, 'plots_{}'.format(model))
    create_dir(dataset_dir)
    if plot_dir is not None:
        dataset_dir = plot_dir
    create_dir(dataset_dir)

    # seed format for title and filename
    print_af = str(af[0])
    if len(af) > 1:
        for s in af:
            if s != af[0]:
                print_af += '_{}'.format(s)

    if type(af) is not list:
        af = [af]

    # Create dataframe with all values
    list_of_df = []
    if type(indicator) is list:
        for i in indicator:
            for a in af:
                _i = i
                list_of_df.append(read_results_json(dataset, model, a, seeds, indicator=_i, unc=unc, indicator_df=True))
    else:
        for a in af:
            list_of_df.append(read_results_json(dataset, model, a, seeds, indicator=indicator))

    df = list_of_df[0]
    for d in range(1, len(list_of_df)):
        df = df.append(list_of_df[d])

    if df.empty: return

    # Create dataframe with 100% data
    full_model_dir = os.path.join(BASE_DIR, 'results')
    # path = os.path.join(full_model_dir, '{}_{}_100%'.format(dataset, model))
    path = os.path.join(RES_DIR, '{}_{}_100%'.format(dataset, model))
    val_acc = []
    test_acc = []
    val_f1 = []
    for seed in seeds:
        all_filepath = os.path.join(path, 'seed_{}_lr_2e-05_bs_32_epochs_5'.format(seed))
        if not os.path.exists(all_filepath): all_filepath = os.path.join(path,
                                                                         'seed_{}_lr_2e-05_bs_32_epochs_20'.format(
                                                                             seed))
        if os.path.exists(all_filepath):
            with open(os.path.join(all_filepath, 'vanilla_results.json')) as json_file:
                results = json.load(json_file)
                val_acc.append(results['val_results']['acc'] * 100)
                test_acc.append(results['test_results']['acc'] * 100)

    # Plot
    if dataset == 'sst-2':
        label_100 = '60.6K training data (100%)'
    elif dataset == 'mrpc':
        label_100 = '3.6K training data (100%)'
    elif dataset == 'qnli':
        label_100 = '105K training data (100%)'
    elif dataset == 'trec-6':
        label_100 = '4.9K training data (100%)'
    elif dataset == 'ag_news':
        label_100 = '114K training data (100%)'
    elif dataset == 'imdb':
        label_100 = '22.5K training data (100%)'
    elif dataset == 'rte':
        label_100 = '2K training data (100%)'
    elif dataset == 'qqp':
        label_100 = '???K training data (100%)'
    elif dataset == 'mnli':
        label_100 = '???K training data (100%)'
    elif dataset == 'dbpedia':
        label_100 = '20K training data'
    if df.empty:
        return

    title_dataset = dataset.upper()
    if title_dataset == 'AG_NEWS':
        title_dataset = 'AGNEWS'

    x = np.linspace(0, int(max(df['samples'])), num=50, endpoint=True)
    x_per = np.linspace(0, int(max(df['data_percent'])), num=50, endpoint=True)

    fig = plt.figure(figsize=(4.0, 3.5))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Change names for plotting:
    # change model names
    df['indicator'].loc[(df['indicator'] == '25_config')] = 'BERT'
    df['indicator'].loc[(df['indicator'] == '25_config_bayes')] = 'BERT+BayesOutput'

    # change df columns
    df = df.rename(columns={'acquisition': 'Acquisition',
                            'indicator': 'Model'})
    if y == 'acc':
        if test:
            y_plot = "test_acc"
        else:
            y_plot = "val_acc"

        # Accuracy
        if val_acc != []:
            df_all = pd.DataFrame()
            df_all['samples'] = x_per
            df_all['test_acc'] = test_acc[0]
            df_all['val_acc'] = val_acc[0]

            # for d in range(1, len(val_acc)):
            for d in range(1, len(test_acc)):
                df_all = df_all.append(pd.DataFrame({'samples': x_per, 'test_acc': test_acc[d], 'val_acc': val_acc[d]}))
            if legend:
                all_ax = sns.lineplot(x="samples", y=y_plot,
                                      data=df_all, ci='sd', estimator='mean', label=label_100,
                                      color='black',
                                      linestyle='-.', legend=False)
            else:
                all_ax = sns.lineplot(x="samples", y=y_plot,
                                      data=df_all, ci='sd', estimator='mean', label=label_100,
                                      color='black',
                                      linestyle='-.')
        if type(indicator) is list:
            al_ax = sns.lineplot(x="data_percent", y=y_plot, hue="Model", style='Acquisition', data=df,
                                 ci='sd',
                                 # ci=None,
                                 # estimator="mean",
                                 estimator="median",
                                 # legend=False,
                                 # palette=sns.color_palette("rocket",3)
                                 )

        plt.xlabel('Acquired dataset size (%)', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)

        if ood:
            plt.title(title_dataset + ' OOD', fontsize=15)
        elif contrast:
            plt.title(title_dataset + ' Contrast set', fontsize=15)
        else:
            plt.title(title_dataset, fontsize=15)

        # plt.legend(loc='lower right', prop={'size': 7})

        # else:
        #     plt.legend(loc='lower right', prop={'size': 7})
        #
        # plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center left', ncol=2,handleheight=1, labelspacing=0.05)
        plt.tight_layout()
        plt.legend(loc='lower right', prop={'size': 10})

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)

        # fix limits
        axes = plt.gca()
        # axes.set_xlim([1, 25])

        # if dataset == 'ag_news':
        #     axes.set_ylim([90, 95])
        #     # start, end = axes.get_ylim()
        #     # axes.yaxis.set_ticks(np.arange(start, end, 2))
        # if dataset == 'imdb':
        #     axes.set_ylim([86, 92])

        plt.style.use("seaborn-colorblind")
        if test:
            print("test")
            filename = "test_acc_{}_{}_{}_{}".format(dataset, model, print_af, indicator)
        else:
            print("val")
            filename = "val_acc_{}_{}_{}_{}".format(dataset, model, print_af, indicator)
        if ood: filename += '_ood'
        if contrast: filename += '_contrast'

        # png
        plt.savefig(os.path.join(dataset_dir, filename + '.png'),
                    dpi=300,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
        # # pdf
        # pp = PdfPages(os.path.join(pdf_dir, filename + '.pdf'))
        # pp.savefig(fig,dpi=300,
        #             transparent=False, bbox_inches="tight", pad_inches=0.1)
        # plt.show()
        # pp.close()
        plt.close()

    return


if __name__ == '__main__':

    datasets = ['imdb', 'rte', 'mrpc', 'qnli', 'sst-2', 'mnli', 'qqp', 'trec-6', 'ag_news']
    seeds = [2, 19, 729, 982, 75]
    indicator = ['small_config', 'small_config_bayes']
    unc = 'vanilla'
    models = ['bert', 'distilbert']

    for dataset in datasets:
        for model in models:
            al_plot(dataset=dataset, indicator=indicator, seeds=seeds, unc=unc, model=model)
