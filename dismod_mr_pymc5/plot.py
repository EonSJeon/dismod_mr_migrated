""" Module for DisMod-MR graphics"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pymc as pm
import arviz as az
import random


colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f0', '#ffff33']


def data_bars(df, style='book', color='black', label=None, max=500):
    """ Plot data bars

    :Parameters:
      - `df` : pandas.DataFrame with columns age_start, age_end, value
      - `style` : str, either book or talk
      - `color` : str, any matplotlib color
      - `label` : str, figure label
      - `max` : int, number of data points to display
    """
    bars = list(zip(df['age_start'], df['age_end'], df['value']))
    if len(bars) > max:
        bars = random.sample(bars, max)

    x, y = [], []
    for a0, a1, v in bars:
        x += [a0, a1, np.nan]
        y += [v, v, np.nan]

    if style == 'book':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=4, color=color, label=label)
    elif style == 'talk':
        # colors 변수는 이미 정의되어 있다고 가정
        plt.plot(x, y, 's-', mew=1, mec='w', ms=0,
                 alpha=1.0, color=colors[2], linewidth=15, label=label)
    else:
        raise ValueError(f'Unrecognized style: {style}')