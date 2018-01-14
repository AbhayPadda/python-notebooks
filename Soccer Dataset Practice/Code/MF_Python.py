%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

midfielder_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, crossing, curve, long_passing, ball_control, acceleration, vision,  balance, stamina FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	midfielder_stats = pd.read_sql_query(midfielder_query, con)

midfielder_stats.head()

midfielder_stats.shape

fig, axs = plt.subplots(1, 5, sharey=True)
midfielder_stats.plot(kind='scatter', x='crossing', y='overall_rating', ax=axs[0], figsize=(16, 8))
midfielder_stats.plot(kind='scatter', x='balance', y='overall_rating', ax=axs[1])
midfielder_stats.plot(kind='scatter', x='stamina', y='overall_rating', ax=axs[2])
midfielder_stats.plot(kind='scatter', x='curve', y='overall_rating', ax=axs[3])
midfielder_stats.plot(kind='scatter', x='dribbling', y='overall_rating', ax=axs[4])

fig, axs1 = plt.subplots(1, 4, sharey=True)
midfielder_stats.plot(kind='scatter', x='long_passing', y='overall_rating', ax=axs1[0], figsize=(16,8))
midfielder_stats.plot(kind='scatter', x='ball_control', y='overall_rating', ax=axs1[1])
midfielder_stats.plot(kind='scatter', x='acceleration', y='overall_rating', ax=axs1[2])
midfielder_stats.plot(kind='scatter', x='vision', y='overall_rating', ax=axs1[3])

import statsmodels.formula.api as smf

# create a fitted model with all the features
lm = smf.ols(formula='overall_rating ~ crossing + dribbling + curve + long_passing + ball_control + acceleration + vision + balance + stamina', data=midfielder_stats).fit()

# print the coefficients
lm.params

lm.summary()

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = midfielder_stats._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()