import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import seaborn as sns

import datetime
matplotlib.use('TkAgg')
# Permet d'afficher l'axe des abscisses plus joliment
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

# Permet d'appliquer le style graphique de Seaborn par défaut sur matplotlib
sns.set()

# La fonction expanduser permet de convertir le répertoire ~ en chemin absolu
data = pd.read_csv(os.path.expanduser("data/sample.csv"))
data["event_time"] = pd.to_datetime(data["event_time"])

data["event_day"] = data["event_time"].dt.day
data["event_hour"] = data["event_time"].dt.hour
data["event_minute"] = data["event_time"].dt.minute
# print(data.head(20))
session = data['user_session'].value_counts() > 1
no_one = data[data['user_session'].isin(session[session].index)].sort_values(by=['user_session'])
no_one = no_one.groupby('user_session')
agg_session = no_one.agg(min=("event_time", np.min), max=("event_time", np.max))
agg_session['time'] = agg_session['max'] - agg_session['min']
# print(agg_session['time'].mean())

filter_hour = agg_session[agg_session['time']<=datetime.timedelta(hours=1)]
# plt.hist(filter_hour['time']/pd.Timedelta(minutes=1), bins=12)
# plt.xlabel("Nombre de minutes")
# plt.ylabel("Nombre de sessions")
# plt.show()
filter_more_hour = agg_session[agg_session['time']>datetime.timedelta(hours=1)]
filter_less_hour = agg_session[agg_session['time']<datetime.timedelta(hours=1)]
# trouver la proportion d'évènement "purchase" dans les évènement de plus d'une heure
print(data[data['user_session'].isin(filter_more_hour.index)]['event_type'].value_counts(normalize=True))