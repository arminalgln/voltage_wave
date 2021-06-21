import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import os
import pickle
#%%
site_files = os.listdir('data/site2/day2/')
site_events = {}
for file in site_files:

    file_path = 'data/site2/day2/' + file
    file_data = scipy.io.loadmat(file_path)['data']
    hour = int(file_data[0][0])

    res = file_data[256:, -1] - file_data[0:-256, -1]
    anomalies = np.union1d(np.where(res > 10), np.where(res < -10))
    res = None
    site_events[hour] = {}
    for ev in anomalies:
        data_row = file_data[ev]
        site_events[hour][ev] = file_data[ev - 2 * 256:ev + 2 * 256, -1]
        fig_path = 'figures/site2/day2_{}_{}_{}_id_{}.png'.format(data_row[0], data_row[1], data_row[2], ev)
        fig = plt.figure()
        plt.ioff()
        plt.plot(site_events[hour][ev])
        plt.interactive(False)
        plt.savefig(fig_path)
        plt.close(fig)


with open("data/events/site2/day2_events.pkl", "wb") as pkl_handle:
	pickle.dump(site_events, pkl_handle)
