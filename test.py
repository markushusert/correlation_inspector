import pandas as pd
import numpy as np
import csv 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib
import math
import correlation_inspector

def load():
    filename="collected_results.csv"
    fields=pd.read_csv(filename, index_col=0, nrows=0).columns.tolist()
    data=np.genfromtxt(filename,skip_header=1,delimiter=",")
    return fields,data.transpose()

fields,data=load()
plt.ion()
correl_inspect=correlation_inspector.correlation_inspector(data,fields)
pass
