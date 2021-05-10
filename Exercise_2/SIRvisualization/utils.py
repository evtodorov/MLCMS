# -*- coding: utf-8 -*-

import os
import pandas as pd
import plotly.graph_objects as go


def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_REMOVED=2):
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted.
    The ID_SUSCEPTIBLE and ID_INFECTED specify which ids the groups have in the Vadere processor file.
    """
    #pedestrian_ids = df['pedestrianId'].unique()
    sim_times = pd.np.arange(0,round(df['simTime'].max()))
    group_counts = pd.DataFrame(columns=['simTime', 'group-s', 'group-i', 'group-r'])
    group_counts['simTime'] = sim_times
    df['simTime_datetime'] = pd.to_datetime(df['simTime'], unit='s') 
    binned_groups = df.groupby([pd.Grouper(key='simTime_datetime',freq='1s'), pd.Grouper('pedestrianId')]).max()
    try:
        valcounts = binned_groups.groupby('simTime_datetime')['groupId-PID5'].value_counts()
    except KeyError: 
        valcounts = binned_groups.groupby('simTime_datetime')['groupId-PID9'].value_counts()
    valcounts.index = valcounts.index.set_levels(valcounts.index.levels[0].astype(int).astype(float)/1000000000, level=0)    
    for name, id in {'group-s':ID_SUSCEPTIBLE, 'group-i':ID_INFECTED, 'group-r': ID_REMOVED}.items():
        try:
            group_counts[name] = valcounts.loc[:,id]
        except:
            group_counts[name] = 0
    return group_counts.fillna(0)


def create_folder_data_scatter(folder):
    """
    Create scatter plot from folder data.
    :param folder:
    :return:
    """
    file_path = os.path.join(folder, "SIRinformation.csv")
    if not os.path.exists(file_path):
        print("wrong name of file")
        return None
    data = pd.read_csv(file_path, delimiter=" ")


    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_REMOVED = 2

    group_counts = file_df_to_count_df(data, ID_INFECTED=ID_INFECTED, ID_SUSCEPTIBLE=ID_SUSCEPTIBLE)
    if group_counts['group-r'].sum() == 0:
    # group_counts.plot()
        scatter_s = go.Scatter(x=group_counts['simTime'],
                               y=group_counts['group-s']+group_counts['group-i'],
                               name='susceptible', 
                               fillcolor = 'rgba(254, 215, 102, 0.7)',
                               line=dict(color='#fed766'),
                               mode='lines', fill='tozeroy')
        scatter_i = go.Scatter(x=group_counts['simTime'],
                               y=group_counts['group-i'],
                               name='infected ', 
                               fillcolor = 'rgba(254, 74, 73, 0.5)',
                               line=dict(color='#fe4a49'),
                               mode='lines', fill='tozeroy')
        return [scatter_s, scatter_i], group_counts
    else:
        scatter_r = go.Scatter(x=group_counts['simTime'],
                               y=group_counts['group-r']+group_counts['group-s']+group_counts['group-i'],
                               name='recovered',
                               fillcolor = 'rgba(42, 183, 202, 0.5)',
                               line=dict(color='#2ab7ca'),
                               mode='lines', fill='tozeroy')
        scatter_s = go.Scatter(x=group_counts['simTime'],
                               y=group_counts['group-s'],
                               name='susceptible ', 
                               fillcolor = 'rgba(254, 215, 102, 0.7)',
                               line=dict(color='#fed766'),
                               mode='lines', fill='tozeroy')
        scatter_i = go.Scatter(x=group_counts['simTime'],
                               y=group_counts['group-i'],
                               name='infected', 
                               fillcolor = 'rgba(254, 74, 73, 0.5)',
                               line=dict(color='#fe4a49'),
                               mode='lines', fill='tozeroy')
        return [scatter_r, scatter_s, scatter_i], group_counts
                               
    
