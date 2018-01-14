get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3


goal_keeper_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	goal_keeper_stats = pd.read_sql_query(goal_keeper_query, con)
    
goal_keeper_stats.shape

fig, axs = plt.subplots(1, 5, sharey=True)
goal_keeper_stats.plot(kind='scatter', x='gk_diving', y='overall_rating', ax=axs[0], figsize=(16, 8))
goal_keeper_stats.plot(kind='scatter', x='gk_handling', y='overall_rating', ax=axs[1])
goal_keeper_stats.plot(kind='scatter', x='gk_kicking', y='overall_rating', ax=axs[2])
goal_keeper_stats.plot(kind='scatter', x='gk_positioning', y='overall_rating', ax=axs[3])
goal_keeper_stats.plot(kind='scatter', x='gk_reflexes', y='overall_rating', ax=axs[4])


from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = goal_keeper_stats._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

goal_keeper_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id WHERE gk_kicking > 69 and gk_reflexes > 69 GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	goal_keeper_stats = pd.read_sql_query(goal_keeper_query, con)
    
import statsmodels.formula.api as smf

# create a fitted model with all the features
lm = smf.ols(formula='overall_rating ~ gk_diving + gk_handling + gk_kicking + gk_positioning + gk_reflexes', data=goal_keeper_stats).fit()

# print the coefficients
lm.params

lm.summary()

