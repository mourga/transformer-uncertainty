import json
import os
import sys

import matplotlib

from src.general import create_dir
from sys_config import RES_DIR, BASE_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                      indicators, methods, task_name, identity_init=False, ece=False,
                      ood=False, few_shot=None, model_type=None):
    df = pd.DataFrame()
    for seed in seeds:
        exp_path = os.path.join(path, 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(seed, learning_rate,
                                                                             per_gpu_train_batch_size,
                                                                             int(num_train_epochs)))
        if identity_init: exp_path += '_identity_init'
        if few_shot is not None: exp_path += '_{}'.format(few_shot)
        if not os.path.exists(exp_path): pass
        df_ = pd.DataFrame()
        for ind in indicators:
            for method in methods:
                res_filename = '{}_results'.format(method)
                if ind is not None: res_filename += '_{}'.format(ind)
                if ood: res_filename += '_ood'
                res_path = os.path.join(exp_path, "{}.json".format(res_filename))
                if not os.path.exists(res_path):
                    pass
                else:
                    with open(res_path) as json_file:
                        results = json.load(json_file)
                        if ece:
                            if ood:
                                val_results_bins = []
                                test_results_bin = results['test_ood_results']['ece']['bins']
                                df0 = pd.DataFrame(
                                    {'val_accs': None,
                                     'val_confs': None,
                                     'test_accs': [a['acc'] if a['acc'] != 0.0 else None for bin, a in
                                                   test_results_bin.items()],
                                     'test_confs': [a['conf'] if a['acc'] != 0.0 else None for bin, a in
                                                    test_results_bin.items()],
                                     'bins': list(np.arange(0.5, 10.5, 1) * 0.1)})
                            else:
                                val_results_bins = results['val_results']['ece']['bins']
                                test_results_bin = results['test_results']['ece']['bins']
                                df0 = pd.DataFrame(
                                    {'val_accs': [a['acc'] if a['acc'] != 0.0 else None for bin, a in
                                                  val_results_bins.items()],
                                     'val_confs': [a['conf'] if a['acc'] != 0.0 else None for bin, a in
                                                   val_results_bins.items()],
                                     'test_accs': [a['acc'] if a['acc'] != 0.0 else None for bin, a in
                                                   test_results_bin.items()],
                                     'test_confs': [a['conf'] if a['acc'] != 0.0 else None for bin, a in
                                                    test_results_bin.items()],
                                     'bins': list(np.arange(0.5, 10.5, 1) * 0.1)})
                            df0['method'] = method
                            if ind is None: _ind = 'Base'
                            if ind == 'adapter': _ind = 'Adapters'
                            if ind == 'bayes_adapter': _ind = 'BayesAdapters'
                            if ind == 'bayes_adapter' and identity_init: _ind = 'BayesAdapters+Identity'
                            if ind == 'bayes_output': _ind = 'BayesOutput'
                            if ind == 'bayes_adapter_bayes_output': _ind = 'BayesAdapters+Output'
                            if ind == 'bayes_adapter_bayes_output_unfreeze': _ind = 'BayesAll'
                            df0['indicator'] = _ind
                            df0['model_type'] = model_type
                        else:
                            if ind is None: _ind = 'Base'
                            if ind == 'adapter': _ind = 'Adapters'
                            if ind == 'bayes_adapter': _ind = 'BayesAdapters'
                            if ind == 'bayes_adapter' and identity_init: _ind = 'BayesAdapters+Identity'
                            if ind == 'bayes_output': _ind = 'BayesOutput'
                            if ind == 'bayes_adapter_bayes_output': _ind = 'BayesAdapters+Output'
                            if ind == 'bayes_adapter_bayes_output_unfreeze': _ind = 'BayesAll'

                            if ood:
                                val_results = None
                                test_results = results['test_ood_results']
                                if 'acc' not in test_results.keys():
                                    val_acc, test_acc = None, None
                                    val_mcc = None
                                    test_mcc = test_results['mcc']
                                else:
                                    val_mcc, test_mcc = None, None
                                    val_acc = None
                                    test_acc = test_results['acc']
                                val_ece, val_nll, val_brier, val_entropy = None, None, None, None
                            else:
                                val_results = results['val_results']
                                test_results = results['test_results']
                                if 'acc' not in val_results.keys():
                                    val_acc, test_acc = None, None
                                    val_mcc = val_results['mcc']
                                    test_mcc = test_results['mcc']
                                else:
                                    val_mcc, test_mcc = None, None
                                    val_acc = val_results['acc']
                                    test_acc = test_results['acc']
                                val_ece = val_results['ece']['ece']
                                val_nll = val_results['nll']['mean']
                                val_entropy = val_results['entropy']['mean']
                                val_brier = val_results['brier']['mean']
                            df0 = pd.DataFrame(
                                {'val_acc': val_acc,
                                 'val_mcc': val_mcc,
                                 'val_ece': val_ece,
                                 'val_nll': val_nll,
                                 'val_entropy': val_entropy,
                                 'val_brier': val_brier,
                                 'test_acc': test_acc,
                                 'test_mcc': test_mcc,
                                 'test_ece': test_results['ece']['ece'],
                                 'test_nll': test_results['nll']['mean'],
                                 'test_entropy': test_results['entropy']['mean'],
                                 'test_brier': test_results['brier']['mean'],
                                 'method': method,
                                 'model_type': model_type,
                                 'indicator': _ind}, index=[0])
                        df_ = df_.append(df0, ignore_index=True)

                    df_['dataset'] = task_name
                    df_['seed'] = seed

        df = df.append(df_, ignore_index=True)
    return df


def uncertainty_plot(task_name, seeds, model_type='bert', learning_rate='2e-05', per_gpu_train_batch_size=16,
                     num_train_epochs='3', indicators=None,
                     methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
                     # methods=["vanilla", "mc3", "mc5", "mc10", "mc20"],
                     identity_init=False,
                     ood=False,
                     few_shot=False):
    path = os.path.join(RES_DIR, '{}_{}_100%'.format(task_name, model_type))
    if not os.path.exists(path): return

    df = pd.DataFrame()

    if type(seeds) is not list: seeds = [seeds]
    if type(indicators) is not list: indicators = [indicators]

    if few_shot:
        df_list = []
        # for f in ['sample_100', 'sample_1000', 'sample_10000']:
        for f in ['sample_20', 'sample_200', 'sample_2000']:
            df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, 20,
                                    indicators, methods, task_name, identity_init=False, ood=ood, few_shot=f)
            df_['num_samples'] = int(f.split('_')[-1])
            df = df.append(df_, ignore_index=True)
    else:
        df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                                indicators, methods, task_name, identity_init=False, ood=ood)
        df = df.append(df_, ignore_index=True)
    # if identity_init:
    #     df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
    #                             indicators, methods, task_name, identity_init=True)
    #     df = df.append(df_, ignore_index=True)

    if not df.empty:

        # for d in ['test', 'val']:
        y_plot = 'acc' if task_name != 'cola' else 'mcc'
        for d in ['test']:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(6.0, 9.0), constrained_layout=True)
            if few_shot:
                df = df[df['method'] == 'vanilla']
                sns.boxplot(x='indicator', y="{}_{}".format(d, y_plot), hue="num_samples", data=df, ax=ax1,
                            palette="deep")
                sns.boxplot(x='indicator', y="{}_ece".format(d), hue="num_samples", data=df, ax=ax2, palette="deep")
                sns.boxplot(x='indicator', y="{}_nll".format(d), hue="num_samples", data=df, ax=ax3, palette="deep")
                sns.boxplot(x='indicator', y="{}_brier".format(d), hue="num_samples", data=df, ax=ax4, palette="deep")
                sns.boxplot(x='indicator', y="{}_entropy".format(d), hue="num_samples", data=df, ax=ax5, palette="deep")
            else:
                sns.boxplot(x='indicator', y="{}_{}".format(d, y_plot), hue="method", data=df, ax=ax1, palette="deep")
                sns.boxplot(x='indicator', y="{}_ece".format(d), hue="method", data=df, ax=ax2, palette="deep")
                sns.boxplot(x='indicator', y="{}_nll".format(d), hue="method", data=df, ax=ax3, palette="deep")
                sns.boxplot(x='indicator', y="{}_brier".format(d), hue="method", data=df, ax=ax4, palette="deep")
                sns.boxplot(x='indicator', y="{}_entropy".format(d), hue="method", data=df, ax=ax5, palette="deep")

            if task_name == "ag_news":
                task_name = 'agnews'

            if ood:
                fig.suptitle(task_name.upper() + ' OOD', fontsize=16)
            elif few_shot:
                fig.suptitle(task_name.upper() + ' Few shot', fontsize=16)
            else:
                fig.suptitle(task_name.upper(), fontsize=16)

            ax1.tick_params(bottom=False)
            ax2.tick_params(bottom=False)
            ax3.tick_params(bottom=False)
            ax4.tick_params(bottom=False)
            ax1.set(xlabel=None)
            ax2.set(xlabel=None)
            ax3.set(xlabel=None)
            ax4.set(xlabel=None)
            ax1.get_legend().remove()
            ax2.get_legend().remove()
            ax3.get_legend().remove()
            ax4.get_legend().remove()

            plt.legend(bbox_to_anchor=(1.4, 1), borderaxespad=0.)

            plt.xlabel('Models', fontsize=14)
            # plt.legend(loc='best', prop={'size': 9})

            plt.style.use("seaborn-colorblind")
            filename = "uncertainty_plot_{}".format(d)
            if identity_init: filename += '_identity'
            if None in indicators:
                filename += '_base'
            else:
                filename += '_adapters'
            if ood: filename += '_ood'
            if few_shot: filename += '_few_shot'
            plt.savefig(os.path.join(path, filename + '.png'),
                        dpi=300,
                        transparent=False, bbox_inches="tight", pad_inches=0.2)

    return


def ece_plot(task_name, seeds, model_type='bert', learning_rate='2e-05', per_gpu_train_batch_size=16,
             num_train_epochs='3', indicators=None, methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
             identity_init=False, plot_method="vanilla",
             ood=False):
    """
    Reliability diagram
    :return:
    """
    path = os.path.join(RES_DIR, '{}_{}_100%'.format(task_name, model_type))
    if not os.path.exists(path): return

    df = pd.DataFrame()

    if type(seeds) is not list: seeds = [seeds]
    if type(indicators) is not list: indicators = [indicators]

    df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                            indicators, methods, task_name, ece=True, ood=ood)
    df = df.append(df_, ignore_index=True)

    if df.empty: return
    d = "test"

    sns.set(style="ticks")
    fig = plt.figure(figsize=(4.0, 3.5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_xlim([0, 1])
    ax.set_xlim([0, 1])

    # plot_method = 'vanilla'
    plotdf = df[df["method"] == plot_method]
    vanilla_df = df[df["method"] == 'vanilla']
    mc_df = df[df["method"] == 'mc10']
    temp_df = df[df["method"] == 'temp_scale']

    base_vanilla = vanilla_df[vanilla_df["indicator"] == 'Base']
    bayes_vanilla = vanilla_df[vanilla_df["indicator"] == 'BayesOutput']
    base_temp = temp_df[temp_df["indicator"] == 'Base']
    bayes_temp = temp_df[temp_df["indicator"] == 'BayesOutput']
    base_mc = mc_df[mc_df["indicator"] == 'Base']
    bayes_mc = mc_df[mc_df["indicator"] == 'BayesOutput']

    # if plot_method == 'temp_scale':
    #     base = plotdf[plotdf["indicator"] == 'Base']
    #     bayes_output = vanilla_df[vanilla_df["indicator"] == 'BayesOutput']
    #     bayes_all = vanilla_df[vanilla_df["indicator"] == 'BayesAll']
    #     base_adapt = plotdf[plotdf["indicator"] == 'Adapters']
    #     base_bayes_adapt = vanilla_df[vanilla_df["indicator"] == 'BayesAdapters']
    #     bayes_adapt_output = vanilla_df[vanilla_df["indicator"] == 'BayesAdapters+Output']
    # else:
    #     base = plotdf[plotdf["indicator"] == 'Base']
    #     bayes_output = plotdf[plotdf["indicator"] == 'BayesOutput']
    #
    #     base_adapt = plotdf[plotdf["indicator"] == 'Adapters']
    #     base_bayes_adapt = plotdf[plotdf["indicator"] == 'BayesAdapters']
    #
    #     bayes_all = plotdf[plotdf["indicator"] == 'BayesAll']
    #     bayes_adapt_output = plotdf[plotdf["indicator"] == 'BayesAdapters+Output']

    # code no error bars
    # plt.plot(base_adapt['test_confs'], base_adapt['test_accs'], ls='-', color='dodgerblue', label='Adapters', marker='o', markersize=4)
    # plt.plot(base_bayes_adapt['test_confs'], base_bayes_adapt['test_accs'], ls='-', color='orangered', label='BayesAdapters', marker='o', markersize=4)
    # plt.plot(base['test_confs'], base['test_accs'], ls='-', color='lime', label='Base', marker='o', markersize=4)
    # plt.plot(bayes_output['test_confs'], bayes_output['test_accs'], ls='-', color='magenta', label='BayesOutput', marker='o', markersize=4)

    # find mean accuracy per bin
    # base_mean = list(base.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # base_std = list(base.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    # base_adapt_mean = list(base_adapt.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # base_adapt_std = list(base_adapt.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    # base_bayes_adapt_mean = list(base_bayes_adapt.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # base_bayes_adapt_std = list(base_bayes_adapt.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    # Bayes Vanilla
    bayes_output_mean = list(bayes_vanilla.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    bayes_output_std = list(bayes_vanilla.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    # Bayes MC
    bayes_mc_output_mean = list(bayes_mc.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    bayes_mc_output_std = list(bayes_mc.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    # # Bayes Vanilla
    # bayes_output_mean = list(bayes_output.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # bayes_output_std = list(bayes_output.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    # bayes_all_mean = list(bayes_all.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # bayes_all_std = list(bayes_all.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    # bayes_adapt_output_mean = list(bayes_adapt_output.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    # bayes_adapt_output_std = list(bayes_adapt_output.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    # Base vanilla
    base_vanilla_mean = list(base_vanilla.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_vanilla_std = list(base_vanilla.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    # Base MC
    base_mc_mean = list(base_mc.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_mc_std = list(base_mc.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    # Base Temperature Scaling
    base_temp_mean = list(base_temp.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_temp_std = list(base_temp.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    bins = list(base_vanilla.groupby('bins', as_index=False)['test_accs'].mean()['bins'])
    if bins == []: bins = list(bayes_vanilla.groupby('bins', as_index=False)['test_accs'].mean()['bins'])

    # Base
    # if not base.empty:
    #     label = 'Base' if plot_method == 'vanilla' else 'Base (Temp. Scaling)'
    #     plt.errorbar(bins, base_mean, base_std, ls='-', color='lime', label=label, marker='o', markersize=4)
    bayes_color = 'orangered'
    base_color_vanilla = 'royalblue'
    base_color_temp = 'navy'
    base_color_mc = 'blue'
    if not base_vanilla.empty:
        # label = 'Base' if plot_method == 'vanilla' else 'Base (Temp. Scaling)'
        label = 'Base (Vanilla)'
        plt.errorbar(bins, base_vanilla_mean, base_vanilla_std, ls='-', color=base_color_vanilla, label=label,
                     marker='o', markersize=3,
                     linewidth=1)
    if not base_mc.empty:
        label = 'Base (MC)'
        plt.errorbar(bins, base_mc_mean, base_mc_std, ls='dashdot', color=base_color_mc, label=label, marker='^',
                     markersize=3,
                     linewidth=1)
    if not base_temp.empty:
        label = 'Base (Temp. Scaling)'
        plt.errorbar(bins, base_temp_mean, base_temp_std, ls='dotted', color=base_color_temp, label=label, marker='v',
                     markersize=3,
                     linewidth=1)

    if not bayes_vanilla.empty:
        # label = 'BayesOutput' if plot_method == 'vanilla' else 'BayesOutput (Vanilla)'
        label = 'BayesOutput (Vanilla)'
        plt.errorbar(bins, bayes_output_mean, bayes_output_std, ls='-', color=bayes_color, label=label, marker='o',
                     markersize=3,
                     linewidth=1)

    if not bayes_mc.empty:
        # label = 'BayesOutput' if plot_method == 'vanilla' else 'BayesOutput (Vanilla)'
        label = 'BayesOutput (MC)'
        plt.errorbar(bins, bayes_mc_output_mean, bayes_mc_output_std, ls='dashdot', color=bayes_color, label=label,
                     marker='^', markersize=3,
                     linewidth=1)
    # if not bayes_all.empty:
    #     plt.errorbar(bins, bayes_all_mean, bayes_all_std, ls='-', color='indigo', label='BayesAll', marker='o', markersize=4)

    # # Adapters
    # if not base_adapt.empty:
    #     if bins == []: bins=list(base_adapt.groupby('bins', as_index=False)['test_accs'].mean()['bins'])
    #     plt.errorbar(bins, base_adapt_mean, base_adapt_std, ls='-', color='dodgerblue', label='Adapters', marker='o', markersize=4)
    # if not base_bayes_adapt.empty:
    #     plt.errorbar(bins, base_bayes_adapt_mean, base_bayes_adapt_std, ls='-', color='orangered', label='BayesAdapters', marker='o', markersize=4)
    # if not bayes_adapt_output.empty:
    #     plt.errorbar(bins, bayes_adapt_output_mean, bayes_adapt_output_std, ls='-', color='yellow', label='BayesOutput', marker='o', markersize=4)

    plt.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), color='black', ls='--')
    plt.legend(loc='upper left', frameon=False, fontsize=9)

    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    if ood:
        plt.title(task_name.upper() + ' OOD')
    else:
        plt.title(task_name.upper())
    plt.style.use("seaborn-colorblind")
    # filename = "reliability_diagram_{}_{}".format(d, plot_method)
    filename = "reliability_diagram_{}_{}".format(d, task_name)
    if identity_init: filename += '_identity'
    # if None in indicators:
    #     filename += '_base'
    # else:
    #     filename += '_adapters'
    if ood: filename += '_ood'
    plt.savefig(os.path.join(path, filename + '.png'),
                dpi=300,
                transparent=False, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    return


def ac_ece_table(datasets, models, indicators, seeds=[2, 19, 729, 982, 75],
                 learning_rate='2e-05',
                 # model_type=model,
                 methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
                 per_gpu_train_batch_size=32,
                 num_train_epochs='5',
                 few_shot=False,
                 ood=False):
    df = pd.DataFrame()
    for task_name in datasets:
        for model_type in models:
            # for ood in [True, False]:
            # for ood in [False]:
            path = os.path.join(RES_DIR, '{}_{}_100%'.format(task_name, model_type))
            # if not os.path.exists(path): return
            # df = pd.DataFrame()

            if type(seeds) is not list: seeds = [seeds]
            # if type(indicators) is not list: indicators = [indicators]

            if few_shot:
                df_list = []
                # for f in ['sample_100', 'sample_1000', 'sample_10000']:
                for f in ['sample_20', 'sample_200', 'sample_2000']:
                    df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, 20,
                                            indicators, methods, task_name, identity_init=False, ood=ood, few_shot=f,
                                            model_type=model_type)
                    df_['num_samples'] = int(f.split('_')[-1])
                    df = df.append(df_, ignore_index=True)
            else:
                df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                                        indicators, methods, task_name, identity_init=False, ood=ood,
                                        model_type=model_type)
                df = df.append(df_, ignore_index=True)
    # df['model_unc'] = df.apply(lambda row: row.indicator + '+' + row.method, axis=1)

    # ID + ECE Vanilla
    df_vanilla = df[df['method'] == 'vanilla']
    df_model = df_vanilla[df_vanilla['indicator'] == 'Base']

    df_table = pd.DataFrame(columns=list(set(df_model.dataset)),
                            index=['bert_acc', 'distilbert_acc', 'bert_ece', 'distilbert_ece'])
    for dataset in list(set(df_model.dataset)):
        df_dataset = df_model[df_model['dataset'] == dataset]
        bert_acc = df_dataset[df_dataset['model_type'] == 'bert']['test_acc'].mean()
        bert_ece = df_dataset[df_dataset['model_type'] == 'bert']['test_ece'].mean()
        distilbert_acc = df_dataset[df_dataset['model_type'] == 'distilbert']['test_acc'].mean()
        distilbert_ece = df_dataset[df_dataset['model_type'] == 'distilbert']['test_ece'].mean()
        df_table[dataset]['bert_acc'] = round(bert_acc, 3) * 100.
        df_table[dataset]['bert_ece'] = round(bert_ece, 3)
        df_table[dataset]['distilbert_acc'] = round(distilbert_acc, 3) * 100.
        df_table[dataset]['distilbert_ece'] = round(distilbert_ece, 3)

    df_table["avg"] = round(df_table.mean(1), 3)
    print()
    path = os.path.join(BASE_DIR, 'paper_results')
    create_dir(path)
    df_table.to_csv(os.path.join(path, 'ac_ece_id.csv'),
                    columns=['imdb', 'sst-2', 'ag_news', 'trec-6', 'qqp', 'mrpc', 'qnli', 'mnli', 'rte', 'avg'])
    return


def reliability_diagram(datasets, models, indicators, seeds=[2, 19, 729, 982, 75],
                        learning_rate='2e-05',
                        # model_type=model,
                        methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
                        per_gpu_train_batch_size=32,
                        num_train_epochs='5',
                        few_shot=False,
                        ood=False,
                        per_method=True,
                        per_dataset=False):
    df = pd.DataFrame()
    for task_name in datasets:
        for model_type in models:
            # for ood in [True, False]:
            # for ood in [False]:
            path = os.path.join(RES_DIR, '{}_{}_100%'.format(task_name, model_type))
            if type(seeds) is not list: seeds = [seeds]
            if type(indicators) is not list: indicators = [indicators]
            if few_shot:
                for f in ['sample_20', 'sample_200', 'sample_2000']:
                    df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, 20,
                                            indicators, methods, task_name, identity_init=False, ood=ood, few_shot=f,
                                            model_type=model_type,
                                            ece=True)
                    df_['num_samples'] = int(f.split('_')[-1])
                    df = df.append(df_, ignore_index=True)
            else:
                df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                                        indicators, methods, task_name, identity_init=False, ood=ood,
                                        model_type=model_type,
                                        ece=True)
                df = df.append(df_, ignore_index=True)

    # only distilbert
    df_distil = df[df['model_type'] == 'distilbert']

    vanilla_df = df_distil[df_distil["method"] == 'vanilla']
    mc_df = df_distil[df_distil["method"] == 'mc10']
    temp_df = df_distil[df_distil["method"] == 'temp_scale']

    base_vanilla = vanilla_df[vanilla_df["indicator"] == 'Base']
    bayes_vanilla = vanilla_df[vanilla_df["indicator"] == 'BayesOutput']
    base_temp = temp_df[temp_df["indicator"] == 'Base']
    bayes_temp = temp_df[temp_df["indicator"] == 'BayesOutput']
    base_mc = mc_df[mc_df["indicator"] == 'Base']
    bayes_mc = mc_df[mc_df["indicator"] == 'BayesOutput']

    if per_method:
        ax2unc = {1: base_vanilla,
                  2: base_mc,
                  3: base_temp,
                  4: bayes_vanilla,
                  5: bayes_mc}

        dataset2color = {'imdb': 'C0',
                         'sst-2': 'C1',
                         'ag_news': 'C2',
                         'trec-6': 'C3',
                         'qqp': 'C4',
                         'mrpc': 'C5',
                         'mnli': 'C6',
                         'qnli': 'C7',
                         'rte': 'C8',
                         }

        # Plot
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(10, 2))

        for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end + 0.01, 0.5))
            ax.xaxis.set_ticks(np.arange(start, end + 0.01, 0.5))
            ax.set_xlabel('Confidence', fontsize=7)
            if i == 0:
                ax.set_ylabel('Accuracy', fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.tick_params(axis='both', which='minor', labelsize=7)
            # diagonal
            ax.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), color='black', ls='--')
            data = ax2unc[i + 1]
            # plot per dataset
            for dataset in list(set(data['dataset'])):
                _data = data[data["dataset"] == dataset]
                bins = list(_data.groupby('bins', as_index=False)['test_accs'].mean()['bins'])
                mean = list(_data.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
                std = list(_data.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
                ax.errorbar(bins, mean, std, ls='-', color=dataset2color[dataset],
                            label=dataset,
                            marker='o',
                            markersize=2,
                            linewidth=1,
                            )

        ax1.set_title('Vanilla', fontsize=7)
        ax2.set_title('MC Dropout', fontsize=7)
        ax3.set_title('Temperature Scaling', fontsize=7)
        ax4.set_title('Bayesian Layer', fontsize=7)
        ax5.set_title('Bayesian Layer + MC', fontsize=7)

        plt.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0., fontsize=8)


        path = os.path.join(BASE_DIR, 'paper_results')
        create_dir(path)
        plt.savefig(os.path.join(path, 'reliability_per_method.png'),
                    dpi=300,
                    transparent=False, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    elif per_dataset:
        method2color = {'Base + vanilla': 'C0',
                         'Base + mc5': 'C1',
                         'Base + temp_scale': 'C2',
                         'BayesOutput + vanilla': 'C3',
                         'BayesOutput + mc5': 'C4',
                         'BayesOutput + temp_scale': 'C5',
                         }
        fig, axs = plt.subplots(1, len(datasets), sharey=True, figsize=(10, 2))

        for i, ax in enumerate(axs):
            dataset=datasets[i]
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end + 0.01, 0.5))
            ax.xaxis.set_ticks(np.arange(start, end + 0.01, 0.5))
            ax.set_xlabel('Confidence', fontsize=7)
            if i == 0:
                ax.set_ylabel('Accuracy', fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.tick_params(axis='both', which='minor', labelsize=7)
            ax.set_title(dataset, fontsize=7)
            # diagonal
            ax.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), color='black', ls='--')
            data = df_distil[df_distil['dataset'] == dataset]
            data['model_unc'] = data.apply(lambda row: row.indicator + ' + ' + row.method, axis=1)

            skip = ["mc3", "mc10", "mc20"]
            uncertainty_methods = [x for x in set(data['model_unc']) if not any(y in x for y in skip)]
            # plot per method
            for method in uncertainty_methods:
                _data = data[data["model_unc"] == method]
                bins = list(_data.groupby('bins', as_index=False)['test_accs'].mean()['bins'])
                mean = list(_data.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
                std = list(_data.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
                ax.errorbar(bins, mean, std, ls='-', color=method2color[method],
                            label=method,
                            marker='o',
                            markersize=2,
                            linewidth=1,
                            )
        plt.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0., fontsize=8)


        path = os.path.join(BASE_DIR, 'paper_results')
        create_dir(path)
        plt.savefig(os.path.join(path, 'reliability_per_dataset.png'),
                    dpi=300,
                    transparent=False, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    return


if __name__ == '__main__':
    # datasets = ['sst-2', 'mrpc', 'qnli', "cola", "mnli", "mnli-mm", "sts-b", "qqp", "rte", "wnli"]
    datasets = ['rte', 'mrpc', 'qnli', 'sst-2', 'mnli', 'qqp', 'trec-6', 'imdb', 'ag_news']
    # datasets = ['rte', 'trec-6']
    models = ['bert', 'distilbert']
    # models = ['distilbert']
    # datasets = ['imdb']

    indicators = [[None, 'bayes_output']]  # ,
    # ['adapter', 'bayes_adapter', 'bayes_adapter_bayes_output']]

    epochs = '5'

    # # (1) ID & ECE TABLE
    # ac_ece_table(
    #     datasets=['imdb', 'sst-2', 'ag_news', 'trec-6', 'qqp', 'mrpc', 'qnli', 'mnli', 'rte'],
    #     models=['bert', 'distilbert'],
    #     indicators=[None, 'bayes_output']
    # )

    # (2) Distil reliability diagram per method
    reliability_diagram(
        # datasets=['imdb', 'sst-2'],
        datasets=['imdb', 'sst-2', 'ag_news', 'trec-6', ],
        # 'qqp', 'mrpc', 'qnli', 'mnli', 'rte'],
        models=['bert', 'distilbert'],
        indicators=[None, 'bayes_output'],
        per_method=False,
        per_dataset=True
    )

    for dataset in datasets:
        print(dataset)
        for ind in indicators:
            for model in models:
    #             print('Plotting uncertainty')
    #             # acc + uncertainty plot
    #             # epochs='20' if dataset == 'trec-6' else '5'
    #             uncertainty_plot(task_name=dataset,
    #                              seeds=[2, 19, 729, 982, 75],
    #                              learning_rate='2e-05',
    #                              model_type=model,
    #                              per_gpu_train_batch_size=32,
    #                              # num_train_epochs='5',
    #                              num_train_epochs=epochs,
    #                              # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
    #                              indicators=ind,
    #                              identity_init=False)
    #             #
                # # Acc + uncertainty OOD
                print('Plotting uncertainty OOD')
                uncertainty_plot(task_name=dataset,
                                 seeds=[2, 19, 729, 982, 75],
                                 learning_rate='2e-05',
                                 model_type=model,
                                 per_gpu_train_batch_size=32,
                                 num_train_epochs='5',
                                 # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
                                 indicators=ind,
                                 ood=True)
    #             #
    #             print('Plotting uncertainty few shot')
    #             uncertainty_plot(task_name=dataset,
    #                              seeds=[2, 19, 729, 982, 75],
    #                              learning_rate='2e-05',
    #                              model_type=model,
    #                              per_gpu_train_batch_size=32,
    #                              num_train_epochs='5',
    #                              # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
    #                              indicators=ind,
    #                              few_shot=True)
    #             #
    #             #
    #             #     print('Plotting reliability diagram (vanilla)')
    #             #     ece_plot(task_name=dataset,
    #             #              seeds=[2, 19, 729, 982, 75],
    #             #              learning_rate='2e-05',
    #             #              per_gpu_train_batch_size=32,
    #             #              # num_train_epochs='5',
    #             #              num_train_epochs=epochs,
    #             #              indicators=ind,
    #             #              identity_init=False, )
    #
    #             print('Plotting reliability diagram (temp scale)')
    #             ece_plot(task_name=dataset,
    #                      seeds=[2, 19, 729, 982, 75],
    #                      learning_rate='2e-05',
    #                      model_type=model,
    #                      per_gpu_train_batch_size=32,
    #                      # num_train_epochs='5',
    #                      num_train_epochs=epochs,
    #                      indicators=ind,
    #                      identity_init=False,
    #                      plot_method='temp_scale')
    #
    #             # print('Plotting reliability diagram OOD (vanilla)')
    #             # ece_plot(task_name=dataset,
    #             #          seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
    #             #          learning_rate='2e-05',
    #             #          per_gpu_train_batch_size=32,
    #             #          num_train_epochs='5',
    #             #          indicators=ind,
    #             #          ood=True, )
