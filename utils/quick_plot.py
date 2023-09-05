import numpy as np
import matplotlib.pyplot as plt

def simple_2d_plot(data, save_dir=None, 
                   title='Plot', xlabel='X', ylabel='Y',
                   x_lim=None, y_lim=None, alpha=0.1):
    '''
    simple 2D plotting
    '''
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1], 'o', markersize=1, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if save_dir is not None:
        plt.savefig(save_dir)    
    return fig, ax