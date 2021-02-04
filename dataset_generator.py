#parametri globali:
#temperatura stanza
#velocita del rullo
#curvatura e lunghezza del mold
#composizione del steel
#presenza di stirrers

#parametri locali:
#temperatura in ogni settore
#pressione dell'acqua
#tipo di ugello

palette = ["#1F77B4","#FF7F0E","#2CA02C", "#00A3E0", '#4943cf', '#1eeca8', '#e52761', '#490b04', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']

from AJ_draw import disegna as ds

import streamlit as st
import numpy as np
import pandas as pd
import random

segments = 13
D_start = 0
D_end = 5000
T_start = 1300
T_end = 1143
vels = [random.uniform(0.7, 1.7) for _ in range(10_000)]

def temperature_simulator(D_start, D_end, T_start, T_end, vel, press, T_room):
    T_start = T_start + T_room - 25
    T_end = T_end + T_room - 25
    #baseline of the temperature
    distance = np.linspace(D_start, D_end, segments, endpoint=True)
    distance = [round(dist,1) for dist in distance]
    def temp_decay(D, T_in, T_out, D_in, D_out):
        return ((D - D_in)/(D_out - D_in))*(T_out - T_in) + T_in
    T_tot = [round(temp_decay(D, T_start, T_end, D_start, D_end),1) for D in distance]

    #adjust the temperature with the velocity
    ang_coef = (T_tot[-1] - T_tot[0])/(distance[-1] - distance[0])*(2-vel)
    def temp_w_vel(vel, D, D_in, T_in):
        return vel*(D - D_in) + T_in
    T_tot_new = [round(temp_w_vel(ang_coef, D, D_start, T_start), 1) for D in distance]

    #adjust the temperature with the preassure of the water
    T_tot_new = np.array(T_tot_new)*(2-np.array(press))
    T_tot_new = np.append(T_tot_new, vel)
    T_tot_new = np.append(T_tot_new, T_room)
    T_tot_new = np.append(T_tot_new, press)
    return T_tot_new, distance

df_temperature = pd.DataFrame()
for vel in vels:
    press = np.random.normal(loc=1, scale=0.02, size=segments)
    press[0] = 1
    T_room = np.random.normal(loc=25, scale=3, size=1)[0]
    temperature, distance = temperature_simulator(D_start, D_end, T_start, T_end, vel, press, T_room)
    df_temperature = pd.concat([df_temperature, pd.DataFrame(temperature).T])

cols = []
temp_cols = []
for i in range(segments):
    cols.append('D'+str(i))
    temp_cols.append('D'+str(i))
cols.append('vel')
cols.append('T_room')
for i in range(segments):
    cols.append('P'+str(i))
df_temperature = df_temperature.reset_index(drop = True)
df_temperature.columns = cols
df_temperature = df_temperature.drop(['P0'], axis = 1)

# st.write(df_temperature)
# ds().nuova_fig()
# for i in range(df_temperature.shape[0]):
#     ds().dati(x = distance, y = df_temperature.iloc[i, 0:13], scat_plot ='scat', colore= palette[i])
# st.pyplot()
# st.write(df_temperature.loc[:,temp_cols])

df_temperature.to_csv('steel_temperature.csv', sep=';')
