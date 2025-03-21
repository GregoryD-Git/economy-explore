# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:13:34 2025

@author: d23gr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# function to plat data
def plot_column(ax, x, y, xlabel, ylabel, line_color, legend_label):
    use_color = colorblind_palette[line_color]
    ax.plot(x, y, color = use_color, label = legend_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def economy_track(days_out, eco_df):
    