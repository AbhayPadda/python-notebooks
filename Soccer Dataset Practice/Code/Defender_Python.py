%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

defender_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, long_passing, long_shots, aggression, interceptions, marking,  standing_tackle, sliding_tackle FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	defender_stats = pd.read_sql_query(defender_query, con)

defender_stats.head()

defender_stats.shape

# Scatter Plots
fig, axs = plt.subplots(1, 4, sharey=True)
defender_stats.plot(kind='scatter', x='long_passing', y='overall_rating', ax=axs[0], figsize=(16, 8))
defender_stats.plot(kind='scatter', x='long_shots', y='overall_rating', ax=axs[1])
defender_stats.plot(kind='scatter', x='aggression', y='overall_rating', ax=axs[2])
defender_stats.plot(kind='scatter', x='interceptions', y='overall_rating', ax=axs[3])

fig, axs1 = plt.subplots(1, 3, sharey=True)
defender_stats.plot(kind='scatter', x='marking', y='overall_rating', ax=axs1[0], figsize=(16,8))
defender_stats.plot(kind='scatter', x='standing_tackle', y='overall_rating', ax=axs1[1])
defender_stats.plot(kind='scatter', x='sliding_tackle', y='overall_rating', ax=axs1[2])

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = defender_stats._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()


defender_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, long_passing, long_shots, aggression, interceptions, marking,  standing_tackle, sliding_tackle FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id WHERE marking > 75 and sliding_tackle > 70 and overall_rating > 80 GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	defender_stats = pd.read_sql_query(defender_query, con)
    
import statsmodels.formula.api as smf

# create a fitted model with all the features
lm = smf.ols(formula='overall_rating ~ long_passing + long_shots + aggression + interceptions + standing_tackle + sliding_tackle', data=defender_stats).fit()

# print the coefficients
lm.params

lm.summary()

