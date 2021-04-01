import json
import os
import sys

import matplotlib

from sys_config import RES_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                      indicators, methods, task_name, identity_init=False, ece=False,
                      ood=False, few_shot=None):
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
                                 'indicator': _ind}, index=[0])
                        df_ = df_.append(df0, ignore_index=True)

                    df_['dataset'] = task_name
                    df_['seed'] = seed

        df = df.append(df_, ignore_index=True)
    return df


def uncertainty_plot(task_name, seeds, model_type='bert', learning_rate='2e-05', per_gpu_train_batch_size=16,
                     num_train_epochs='3', indicators=None,
                     methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
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
        for f in ['sample_100', 'sample_1000', 'sample_10000']:
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
                df=df[df['method']=='vanilla']
                sns.boxplot(x='indicator', y="{}_{}".format(d, y_plot), hue="num_samples", data=df, ax=ax1, palette="deep")
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
                            indicators, methods, task_name, ece=True,ood=ood)
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

    if plot_method == 'temp_scale':
        base = plotdf[plotdf["indicator"] == 'Base']
        bayes_output = vanilla_df[vanilla_df["indicator"] == 'BayesOutput']
        bayes_all = vanilla_df[vanilla_df["indicator"] == 'BayesAll']
        base_adapt = plotdf[plotdf["indicator"] == 'Adapters']
        base_bayes_adapt = vanilla_df[vanilla_df["indicator"] == 'BayesAdapters']
        bayes_adapt_output = vanilla_df[vanilla_df["indicator"] == 'BayesAdapters+Output']
    else:
        base = plotdf[plotdf["indicator"] == 'Base']
        bayes_output = plotdf[plotdf["indicator"] == 'BayesOutput']

        base_adapt = plotdf[plotdf["indicator"] == 'Adapters']
        base_bayes_adapt = plotdf[plotdf["indicator"] == 'BayesAdapters']

        bayes_all = plotdf[plotdf["indicator"] == 'BayesAll']
        bayes_adapt_output = plotdf[plotdf["indicator"] == 'BayesAdapters+Output']

    # code no error bars
    # plt.plot(base_adapt['test_confs'], base_adapt['test_accs'], ls='-', color='dodgerblue', label='Adapters', marker='o', markersize=4)
    # plt.plot(base_bayes_adapt['test_confs'], base_bayes_adapt['test_accs'], ls='-', color='orangered', label='BayesAdapters', marker='o', markersize=4)
    # plt.plot(base['test_confs'], base['test_accs'], ls='-', color='lime', label='Base', marker='o', markersize=4)
    # plt.plot(bayes_output['test_confs'], bayes_output['test_accs'], ls='-', color='magenta', label='BayesOutput', marker='o', markersize=4)

    # find mean accuracy per bin
    base_mean = list(base.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_std = list(base.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    base_adapt_mean = list(base_adapt.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_adapt_std = list(base_adapt.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    base_bayes_adapt_mean = list(base_bayes_adapt.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    base_bayes_adapt_std = list(base_bayes_adapt.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    bayes_output_mean = list(bayes_output.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    bayes_output_std = list(bayes_output.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    bayes_all_mean = list(bayes_all.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    bayes_all_std = list(bayes_all.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])
    bayes_adapt_output_mean = list(bayes_adapt_output.groupby('bins', as_index=False)['test_accs'].mean()['test_accs'])
    bayes_adapt_output_std = list(bayes_adapt_output.groupby('bins', as_index=False)['test_accs'].std()['test_accs'])

    bins = list(base.groupby('bins', as_index=False)['test_accs'].mean()['bins'])

    # Base
    if not base.empty:
        label = 'Base' if plot_method == 'vanilla' else 'Base (Temp. Scaling)'
        plt.errorbar(bins, base_mean, base_std, ls='-', color='lime', label=label, marker='o', markersize=4)
    if not bayes_output.empty:
        label = 'BayesOutput' if plot_method == 'vanilla' else 'BayesOutput (Vanilla)'
        plt.errorbar(bins, bayes_output_mean, bayes_output_std, ls='-', color='magenta', label=label, marker='o', markersize=4)
    if not bayes_all.empty:
        plt.errorbar(bins, bayes_all_mean, bayes_all_std, ls='-', color='indigo', label='BayesAll', marker='o', markersize=4)

    # Adapters
    if not base_adapt.empty:
        if bins == []: bins=list(base_adapt.groupby('bins', as_index=False)['test_accs'].mean()['bins'])
        plt.errorbar(bins, base_adapt_mean, base_adapt_std, ls='-', color='dodgerblue', label='Adapters', marker='o', markersize=4)
    if not base_bayes_adapt.empty:
        plt.errorbar(bins, base_bayes_adapt_mean, base_bayes_adapt_std, ls='-', color='orangered', label='BayesAdapters', marker='o', markersize=4)
    if not bayes_adapt_output.empty:
        plt.errorbar(bins, bayes_adapt_output_mean, bayes_adapt_output_std, ls='-', color='yellow', label='BayesOutput', marker='o', markersize=4)

    plt.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), color='black', ls='--')
    plt.legend(loc='upper left', frameon=False)

    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    if ood:
        plt.title(task_name.upper()+' OOD')
    else:
        plt.title(task_name.upper())
    plt.style.use("seaborn-colorblind")
    filename = "reliability_diagram_{}_{}".format(d, plot_method)
    if identity_init: filename += '_identity'
    if None in indicators:
        filename += '_base'
    else:
        filename += '_adapters'
    if ood: filename += '_ood'
    plt.savefig(os.path.join(path, filename + '.png'),
                dpi=300,
                transparent=False, bbox_inches="tight", pad_inches=0.2)

    return


if __name__ == '__main__':

    # datasets = ['sst-2', 'mrpc', 'qnli', "cola", "mnli", "mnli-mm", "sts-b", "qqp", "rte", "wnli"]
    datasets = ['rte', 'mrpc', 'qnli', 'sst-2', 'cola', 'mnli', 'qqp', 'trec-6']
    # datasets = ['sst-2', 'mnli', 'qqp']

    indicators = [[None, 'bayes_output']]#,
                  # ['adapter', 'bayes_adapter', 'bayes_adapter_bayes_output']]

    epochs = '5'
    for dataset in datasets:
        print(dataset)
        for ind in indicators:
            print('Plotting uncertainty')
            # acc + uncertainty plot
            epochs='20' if dataset == 'trec-6' else '5'
            uncertainty_plot(task_name=dataset,
                             seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                             learning_rate='2e-05',
                             per_gpu_train_batch_size=32,
                             # num_train_epochs='5',
                             num_train_epochs=epochs,
                             # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
                             indicators=ind,
                             identity_init=False)

            # # Acc + uncertainty OOD
            print('Plotting uncertainty OOD')
            uncertainty_plot(task_name=dataset,
                             seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                             learning_rate='2e-05',
                             per_gpu_train_batch_size=32,
                             num_train_epochs='5',
                             # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
                             indicators=ind,
                             ood=True)

            print('Plotting uncertainty few shot')
            uncertainty_plot(task_name=dataset,
                             seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                             learning_rate='2e-05',
                             per_gpu_train_batch_size=32,
                             num_train_epochs='5',
                             # indicators=[None, 'adapter', 'bayes_adapter', 'bayes_output'],
                             indicators=ind,
                             few_shot=True)


            print('Plotting reliability diagram (vanilla)')
            ece_plot(task_name=dataset,
                     seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                     learning_rate='2e-05',
                     per_gpu_train_batch_size=32,
                     # num_train_epochs='5',
                     num_train_epochs=epochs,
                     indicators=ind,
                     identity_init=False, )

            print('Plotting reliability diagram (temp scale)')
            ece_plot(task_name=dataset,
                     seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                     learning_rate='2e-05',
                     per_gpu_train_batch_size=32,
                     # num_train_epochs='5',
                     num_train_epochs=epochs,
                     indicators=ind,
                     identity_init=False,
                     plot_method='temp_scale')

            print('Plotting reliability diagram OOD (vanilla)')
            ece_plot(task_name=dataset,
                     seeds=[2, 19, 729, 982, 75, 281, 325, 195, 83, 4],
                     learning_rate='2e-05',
                     per_gpu_train_batch_size=32,
                     num_train_epochs='5',
                     indicators=ind,
                     ood=True, )
