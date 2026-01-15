import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

"""
Functions to help with plotting optimisation results, such as heatmaps and bar charts.
"""

def produce_heatmap(result_arr, x_arr, y_arr, x_label='x', y_label='y', z_label='value', title_str='Heatmap'):
    """
    Produces a heatmap from a 2D array of results.

    Parameters:
        result_arr (2D array): Array of results to plot.
        x_arr (1D array): Array of x-axis values.
        y_arr (1D array): Array of y-axis values.
        x_label (str): description of x-axis values
        y_label (str): description of y-axis values
        z_label (str): description of result_arr values
        title_str (str): Title of the heatmap.
    """
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    plt.imshow(result_arr.T, 
            extent=[x_arr.min()-dx/2, x_arr.max()+dx/2, 
                    y_arr.min()-dy/2, y_arr.max()+dy/2],
            origin='lower', 
            aspect='auto', 
            cmap='viridis',)
	
    plt.colorbar(label=z_label)
    plt.xlabel(x_label)
    plt.xticks(x_arr)
    plt.yticks(y_arr)
    plt.ylabel(y_label)
    plt.title(title_str)
    plt.legend()



def plot_pretty_bar_chart(dict_arr, label_arr, title=""):
    """
    Creates a grouped bar chart from an array of dictionaries. Used for visualizing tariff compositions.
    Parameters:
        dict_arr (list of dict): Each dictionary contains category-value pairs for a specific tariff structure.
        label_arr (list of str): Labels corresponding to each dictionary in dict_arr.
        title (str): Title of the bar chart.
    """
    dict_arr = deepcopy(dict_arr)
    label_arr = deepcopy(label_arr)

    labels = list(dict_arr[0].keys())
    labels.append("total")

    x = np.arange(len(labels))   # category positions
    l = len(dict_arr)
    width = 1/(l+1)  # width of the bars 


    l_arr = np.arange(0.5-l/2, 0.5+l/2)

    for index, i in enumerate(l_arr):
        dict_arr[index]["total"] = sum(dict_arr[index].values())
        plt.bar(x+i*width, list(dict_arr[index].values()), width=width, label=label_arr[index])

    plt.ylabel("Tariff Contribution (MYR)")
    if title!="":
        plt.title(title)
    else:
        plt.title("Tariff compostion under Different Tariff Structures")

    plt.xticks(x, rotation=20, ha="right", labels=labels)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend()