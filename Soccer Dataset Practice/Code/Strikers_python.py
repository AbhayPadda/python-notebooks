%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

striker_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, finishing, heading_accuracy, sprint_speed, shot_power, jumping, penalties, short_passing, acceleration, vision, positioning, volleys, dribbling, ball_control, agility, reactions, stamina FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	striker_stats = pd.read_sql_query(striker_query, con)

striker_stats.shape

fig, axs = plt.subplots(1, 4, sharey=True)
striker_stats.plot(kind='scatter', x='penalties', y='overall_rating', ax=axs[0], figsize=(16, 8))
striker_stats.plot(kind='scatter', x='finishing', y='overall_rating', ax=axs[1])
striker_stats.plot(kind='scatter', x='heading_accuracy', y='overall_rating', ax=axs[2])
striker_stats.plot(kind='scatter', x='sprint_speed', y='overall_rating', ax=axs[3])

fig, axs1 = plt.subplots(1, 4, sharey=True)
striker_stats.plot(kind='scatter', x='shot_power', y='overall_rating', ax=axs1[0], figsize=(16,8))
striker_stats.plot(kind='scatter', x='jumping', y='overall_rating', ax=axs1[1])
striker_stats.plot(kind='scatter', x='penalties', y='overall_rating', ax=axs1[2])
striker_stats.plot(kind='scatter', x='short_passing', y='overall_rating', ax=axs1[3])

fig, axs2 = plt.subplots(1, 4, sharey=True)
striker_stats.plot(kind='scatter', x='acceleration', y='overall_rating', ax=axs2[0], figsize=(16,8))
striker_stats.plot(kind='scatter', x='vision', y='overall_rating', ax=axs2[1])
striker_stats.plot(kind='scatter', x='positioning', y='overall_rating', ax=axs2[2])
striker_stats.plot(kind='scatter', x='volleys', y='overall_rating', ax=axs2[3])

fig, axs2 = plt.subplots(1, 5, sharey=True)
striker_stats.plot(kind='scatter', x='dribbling', y='overall_rating', ax=axs2[0], figsize=(16,8))
striker_stats.plot(kind='scatter', x='ball_control', y='overall_rating', ax=axs2[1])
striker_stats.plot(kind='scatter', x='agility', y='overall_rating', ax=axs2[2])
striker_stats.plot(kind='scatter', x='reactions', y='overall_rating', ax=axs2[3])
striker_stats.plot(kind='scatter', x='stamina', y='overall_rating', ax=axs2[4])


striker_stats.head()
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = striker_stats._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()



striker_query = "SELECT DATE(MAX(date_stat)), Player_name, overall_rating, finishing, heading_accuracy, sprint_speed, shot_power, jumping, penalties, short_passing, acceleration, vision, positioning, volleys, dribbling, ball_control, agility, reactions, stamina FROM Player_Stats ps JOIN Player p ON p.player_api_id = ps.player_api_id WHERE finishing > 75 GROUP BY ps.player_api_id"

with sqlite3.connect('E:/Abhay MBA Docs/TERM 1/Programming for analysts/Group Project/soccer/database.sqlite') as con:
	striker_stats = pd.read_sql_query(striker_query, con)

import statsmodels.formula.api as smf

# create a fitted model with all the features
lm = smf.ols(formula='overall_rating ~ penalties + finishing + heading_accuracy + sprint_speed + shot_power + jumping + penalties + short_passing + acceleration + vision + positioning + volleys + dribbling + ball_control + agility + reactions + stamina', data=striker_stats).fit()

# print the coefficients
lm.params

lm.summary()