import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import pandas as pd
from scipy.stats import gaussian_kde

with open('metrics.pickle', 'rb') as handle:
    metrics = pickle.load(handle)

with open('lengths.pickle', 'rb') as handle:
    lengths = pickle.load(handle)

"""fig, ax = plt.subplots()
x = np.array(lengths)*4
y = metrics[0]['precision'][0]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
ax.scatter(x, y, c=z, s=200*z)
#ax.set_ylim([0, 0.2])
#ax.set_xlim([0, 500])
plt.show()
fig, ax = plt.subplots()
x = np.array(lengths)*4
y = metrics[0]['ndcg'][0]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
ax.scatter(x, y, c=z, s=20)
#ax.set_ylim([0, 0.2])
#ax.set_xlim([0, 500])
plt.show()
fig, ax = plt.subplots()
x = np.array(lengths)*4
y = metrics[0]['recall'][0]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
ax.scatter(x, y, c=z, s=20)
#ax.set_ylim([0, 0.2])
#ax.set_xlim([0, 500])
plt.show()
print("Done")

data = {"movies seen": np.array(lengths)*4, "precision": metrics[0]['precision'][0]}
df = pd.DataFrame(data)
fig = px.density_heatmap(df, x="movies seen", y="precision", nbinsx=50, nbinsy=50, color_continuous_scale="Viridis")
fig.show()

data = {"movies seen": np.array(lengths)*4, "recall": metrics[0]['recall'][0]}
df = pd.DataFrame(data)
fig = px.density_heatmap(df, x="movies seen", y="recall", nbinsx=50, nbinsy=50, color_continuous_scale="Viridis")
fig.show()

data = {"movies seen": np.array(lengths)*4, "ndcg": metrics[0]['ndcg'][0]}
df = pd.DataFrame(data)
fig = px.density_heatmap(df, x="movies seen", y="ndcg", nbinsx=50, nbinsy=50, color_continuous_scale="Viridis")
fig.show()
"""

data = {"movies seen": np.array(lengths)*4, "precision": metrics[0]['precision'][0]}
df = pd.DataFrame(data)
bins = np.arange(-1, 2000, 1)
labels = np.arange(0, 2000, 1)
count = df['precision'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).count()
df = df['precision'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).mean()
df.to_frame().reset_index().plot(x="movies seen", y="precision", kind="scatter", s=count, label="number of users")
plt.savefig("plots/precision_hist.png")
plt.show()


data = {"movies seen": np.array(lengths)*4, "recall": metrics[0]['recall'][0]}
df = pd.DataFrame(data)
bins = np.arange(-1, 2000, 1)
labels = np.arange(0, 2000, 1)
count = df['recall'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).count()
df = df['recall'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).mean()
df.to_frame().reset_index().plot(x="movies seen", y="recall", kind="scatter", s=count, label="number of users")
plt.savefig("plots/recall_hist.png")
plt.show()


data = {"movies seen": np.array(lengths)*4, "ndcg": metrics[0]['ndcg'][0]}
df = pd.DataFrame(data)
bins = np.arange(-1, 2000, 1)
labels = np.arange(0, 2000, 1)
count = df['ndcg'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).count()
df = df['ndcg'].groupby(pd.cut(df['movies seen'], bins=bins, labels=labels)).mean()
df.to_frame().reset_index().plot(x="movies seen", y="ndcg", kind="scatter", s=count, label="number of users")
plt.savefig("plots/ndcg_hist.png")
plt.show()
print("Done")