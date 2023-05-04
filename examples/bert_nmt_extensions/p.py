import pandas as pd
import numpy as np
import os
# one cell
root="/hy-tmp/fairseq/ckpt/exp_bertattn/logs/"
names=["base","10", "noenc", "nodec", "share_enc"]
names=["base","10", "noenc", "share_enc"]

def get_filename(root,name):
    return os.path.join(root,f"ppl_{name}.txt")

dfs = [pd.read_csv(get_filename(root,name), header=None, names=["ppl"]) for name in names]

import matplotlib.pyplot as plt
# x = np.linspace(0, 2, 100)  # Sample data.
x = list(range(len(dfs[0])))
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(5, 2.7))
print(names)
for name,df in zip(names, dfs):
    ax.plot(x, df["ppl"], label=name)  # Plot some data on the axes.
ax.set_xlabel('epoch')  # Add an x-label to the axes.
ax.set_ylabel('ppl')  # Add a y-label to the axes.
ax.set_title("PPL")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()