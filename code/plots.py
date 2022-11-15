import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

folders = os.listdir("plots")
folders = [f for f in folders if "." not in f]

for folder in folders:
    files = os.listdir(os.path.join("plots", folder))
    files = sorted([f for f in files if ".DS" not in f])
    df = pd.read_csv(os.path.join("plots", folder, files[0]))[["Step", "Value"]]
    ax = df.plot(x="Step", title=folder)
    for file in files[1:]:
        df = pd.read_csv(os.path.join("plots", folder, file))[["Step", "Value"]]
        df.plot(x="Step", ax=ax)

    labels = [file.split("_")[0].split(".")[0] for file in files]
    ax.legend(labels)
    plt.savefig("plots/"+ folder +".png")


