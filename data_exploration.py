import pandas as pd
import os


appl_train = pd.read_csv("data/application_train.csv")
appl_test = pd.read_csv("data/application_test.csv")
df = pd.read_csv('data/data_full.csv', index_col=[0])
appl_train.TARGET.replace({0:False,
                           1:True}).value_counts(normalize=True).plot(kind='bar',
                                                                      title='Payment default repartition')


(counts / len(df)).plot(kind='bar')
plt.xticks([], [])
# plt.savefig('plots/valeurs_manquantes.png', bbox_inches='tight', dpi=720)

