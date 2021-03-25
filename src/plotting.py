import json
import os
import sys

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from src.general import create_dir
from sys_config import RES_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from sys_config import FIG_DIR
# from utilities.general import create_dir

def read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                      indicators, methods, task_name, identity_init=False):
    df = pd.DataFrame()
    for seed in seeds:
        exp_path = os.path.join(path, 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(seed, learning_rate,
                                                                            per_gpu_train_batch_size,
                                                                            int(num_train_epochs)))
        if identity_init: exp_path += '_identity_init'
        if not os.path.exists(exp_path): pass
        df_ = pd.DataFrame()
        for ind in indicators:
            for method in methods:
                res_filename = '{}_results'.format(method)
                if ind is not None: res_filename += '_{}'.format(ind)
                res_path = os.path.join(exp_path, "{}.json".format(res_filename))
                if not os.path.exists(res_path):
                    pass
                else:
                    with open(res_path) as json_file:
                        results = json.load(json_file)
                        if ind is None: _ind = 'Base'
                        if ind=='adapter': _ind = 'Adapters'
                        if ind=='bayes_adapter': _ind = 'BayesAdapters'
                        if ind=='bayes_adapter' and identity_init: _ind = 'BayesAdapters+Identity'

                        val_results = results['val_results']
                        test_results = results['test_results']
                        df0 = pd.DataFrame(
                            {'val_acc': val_results['acc'],
                             'val_ece': val_results['ece']['ece'],
                             'val_nll': val_results['nll']['mean'],
                             'val_entropy': val_results['entropy']['mean'],
                             'val_brier': val_results['brier']['mean'],
                             'test_acc': test_results['acc'],
                             'test_ece': test_results['ece']['ece'],
                             'test_nll': test_results['nll']['mean'],
                             'test_entropy': test_results['entropy']['mean'],
                             'test_brier': test_results['brier']['mean'],
                             'method': method,
                             'indicator':_ind}, index=[0])
                        df_ = df_.append(df0, ignore_index=True)

                    # df_['val_acc'] = val_acc
                    df_['dataset'] = task_name
                    df_['seed'] = seed
                    # df_['num_samples'] = num_samples

        df = df.append(df_, ignore_index=True)
    return df

def uncertainty_plot(task_name, seeds, model_type='bert', learning_rate='2e-05', per_gpu_train_batch_size=16,
                     num_train_epochs='3', indicators=None, methods=["vanilla", "mc3", "mc5", "mc10", "mc20", "temp_scale"],
                     identity_init=False):

    path = os.path.join(RES_DIR, '{}_{}_100%'.format(task_name, model_type))
    if not os.path.exists(path): return

    df = pd.DataFrame()

    if type(seeds) is not list: seeds = [seeds]
    if type(indicators) is not list: indicators = [indicators]

    df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                            indicators, methods, task_name, identity_init=False)
    df = df.append(df_, ignore_index=True)
    if identity_init:
        df_ = read_results_json(seeds, path, learning_rate, per_gpu_train_batch_size, num_train_epochs,
                            indicators, methods, task_name, identity_init=True)
        df = df.append(df_, ignore_index=True)
    # for seed in seeds:
    #     exp_path = os.path.join(path, 'seed_{}_lr_{}_bs_{}_epochs_{}'.format(seed, learning_rate,
    #                                                                         per_gpu_train_batch_size,
    #                                                                         int(num_train_epochs)))
    #     if identity_init: exp_path += '_identity_init'
    #     if not os.path.exists(exp_path): pass
    #     df_ = pd.DataFrame()
    #     for ind in indicators:
    #         for method in methods:
    #             res_filename = '{}_results'.format(method)
    #             if ind is not None: res_filename += '_{}'.format(ind)
    #             res_path = os.path.join(exp_path, "{}.json".format(res_filename))
    #             if not os.path.exists(res_path):
    #                 pass
    #             else:
    #                 with open(res_path) as json_file:
    #                     results = json.load(json_file)
    #                     if ind is None: _ind = 'Base'
    #                     if ind=='adapter': _ind = 'Base+Adapters'
    #                     if ind=='bayes_adapter': _ind = 'Base+BayesAdapters'
    #
    #                     val_results = results['val_results']
    #                     test_results = results['test_results']
    #                     df0 = pd.DataFrame(
    #                         {'val_acc': val_results['acc'],
    #                          'val_ece': val_results['ece']['ece'],
    #                          'val_nll': val_results['nll']['mean'],
    #                          'val_entropy': val_results['entropy']['mean'],
    #                          'val_brier': val_results['brier']['mean'],
    #                          'test_acc': test_results['acc'],
    #                          'test_ece': test_results['ece']['ece'],
    #                          'test_nll': test_results['nll']['mean'],
    #                          'test_entropy': test_results['entropy']['mean'],
    #                          'test_brier': test_results['brier']['mean'],
    #                          'method': method,
    #                          'indicator':_ind}, index=[0])
    #                     df_ = df_.append(df0, ignore_index=True)
    #
    #                 # df_['val_acc'] = val_acc
    #                 df_['dataset'] = task_name
    #                 df_['seed'] = seed
    #                 # df_['num_samples'] = num_samples
    #
    #     df = df.append(df_, ignore_index=True)
    # df = df_
    if not df.empty:
        # df['indicator'].loc[(df['indicator'] == None)] = 'Base'

        for d in ['test', 'val']:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(6.0, 9.0), constrained_layout=True)
            # sns.boxplot(x="num_samples", y="ece", hue="method", data=df, ax=ax1, palette=sns.color_palette("hls", 5))
            sns.boxplot(x='indicator', y="{}_acc".format(d), hue="method", data=df, ax=ax1, palette="deep")
            sns.boxplot(x='indicator', y="{}_ece".format(d), hue="method", data=df, ax=ax2, palette="deep")
            # sns.boxplot(x="num_samples", y="nll", hue="method", data=df, ax=ax2, palette=sns.color_palette("hls", 5))
            sns.boxplot(x='indicator', y="{}_nll".format(d), hue="method", data=df, ax=ax3, palette="deep")
            # sns.boxplot(x="num_samples", y="brier", hue="method", data=df, ax=ax3, palette=sns.color_palette("hls", 5))
            sns.boxplot(x='indicator', y="{}_brier".format(d), hue="method", data=df, ax=ax4, palette="deep")
            # sns.boxplot(x="num_samples", y="entropy", hue="method", data=df, ax=ax4, palette=sns.color_palette("hls", 5))
            sns.boxplot(x='indicator', y="{}_entropy".format(d), hue="method", data=df, ax=ax5, palette="deep")

            if task_name == "ag_news":
                task_name = 'agnews'
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

            # unc_dir = os.path.join(ex, 'uncertainty_plots')
            # create_dir(unc_dir)
            plt.savefig(os.path.join(path, filename + '.png'),
                        dpi=300,
                        transparent=False, bbox_inches="tight", pad_inches=0.2)

            # pdf
            # pp = PdfPages(os.path.join(exp_path, filename + '.pdf'))
            # pp.savefig(fig)
            # pp.close()
    return

if __name__ == '__main__':

    # uncertainty plot
    # datasets = ['sst-2', 'mrpc', 'qnli', "cola", "mnli", "mnli-mm", "sts-b", "qqp", "rte", "wnli"]
    datasets = ['rte', 'mrpc', 'sst-2', 'qnli']

    epochs='5'
    for dataset in datasets:
        # pass
        uncertainty_plot(task_name=dataset,
                         seeds=[2,19,729,982, 75, 281, 325, 195, 83, 4],
                         learning_rate='2e-05',
                         per_gpu_train_batch_size=32,
                         num_train_epochs='5',
                         indicators=[None, 'adapter', 'bayes_adapter'],
                         identity_init=True)