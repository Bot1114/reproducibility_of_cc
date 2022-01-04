import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('resultFig3.csv')
pos_sam = df['positive_samples'].values
neg_sam = df['negative_samples'].values
pos_clu = df['positive_clusters'].values
neg_clu = df['negative_clusters'].values
names = ['pos. sample', 'neg. sample', 'pos. cluster', 'neg. cluster']
# plt.title('')
# plt.title('CC on CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Similarity')
plt.plot(pos_sam)
plt.plot(neg_sam)
plt.plot(pos_clu, linestyle="dashed")
plt.plot(neg_clu, linestyle="dashed")
plt.legend(names, loc="right")
plt.show()