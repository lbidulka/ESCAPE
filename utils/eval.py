import numpy as np
import torch

import matplotlib.pyplot as plt


def plot_errs_err(gt_err, s_err, out_path):
    '''
    Plots err signal vs true error

    args:
        gt_err: error btw backbone pred and gt
        err_s: predicted error
        out_path: path to save plot
    '''

    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(spec[0, 0])

    ax1.scatter(gt_err, s_err, s=1)
    ax1.set_xlabel('gt_err')
    ax1.set_ylabel('s_err')
    ax1.set_title('s_err vs gt_err')

    ax1.set_xlim([0, 3.0])
    ax1.set_ylim([0, 0.01])

    fig.savefig(out_path)
    plt.close(fig)

