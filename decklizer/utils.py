import numpy as np
import pandas as pd
import math
import urllib
import copy
import base64
import io
import re
import datetime
import time
from random import shuffle, choice
import urllib
from os.path import dirname, join
# from engine import *

def load_schedule(df=None,
                  customer = 'AHP',
                  technology = 'SM',
                  color = 'WHITE',
                  cycle = 'CYCLE 1'):
    if df is None:
        module_path = dirname(__file__)
        with open(join(module_path, 'data', '200721_ New Way SCH W3-W6 W14 07.20.20.xlsx'), 'rb') as \
                excel_file:
            df = pd.read_excel(excel_file,
                               sheet_name='Schedule')


    df_filtered = df.loc[df['Customer Name'] == customer]
    df_filtered = df_filtered.loc[df_filtered['Description'].str.contains(technology)]
    df_filtered = df_filtered.loc[df_filtered['Description'].str.contains(color)] #CYAN, TEAL
    df_filtered = df_filtered.loc[df_filtered['CYCLE / BUCKET'] == cycle]
    df_filtered = df_filtered.loc[df_filtered['Total LM Order QTY'] > 0]
    df_filtered.insert(0, 'Block', 1)
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['Order Number'] = df_filtered.index + 1

    df_filtered['Width'] = pd.DataFrame(list(pd.DataFrame(list(df_filtered['Description']
                                                      .str.split(';')))[0].str.split('UN0')))[1]
    return df_filtered

def process_schedule(df_filtered, B=4160, put_up=17000, doffs_in_jumbo=6, verbiose=True):
    lm = list(df_filtered.groupby('Description')['Total LM Order QTY'].sum().values)
    widths = list(df_filtered.groupby('Width')['Total LM Order QTY'].sum().index.values.astype(int))
    L = doff_length = put_up * doffs_in_jumbo # df['LM putup']
    neckins = []
    for width in widths:
        if width < 170:
            neckin = 4
        elif width < 208:
            neckin = 5
        else:
            neckin = 7
        neckins.append(neckin)
    w = list(np.array(widths) + np.array(neckins)) # the values used in the actual computation
    q = [math.ceil(x/L) for x in lm] # jumbos needed per width, rounded up
    # s = BinPackingExample(w, q) # material orders (list of widths, 1 width = 1 jumbo)
    if verbiose:
        print('The important variables', end='\n\n')
        print('widths: {} (mm)'.format(widths))
        print('neck in: {} (mm)'.format(neckins))
        print('jumbo length (L): {} (m)'.format(L))
        print('undeckled jumbos needed (q): {}'.format(q))
    return w, q, L, neckins

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
    return df.to_json()

def parse_description(text, grab='tech'):
    '''
    returns technology type from schedule description
    '''
    if type(text) == str:
        match = re.match(r"([a-z]+)([0-9]+)([a-z]+0)([0-9]+)", text, re.I)
        if match:
            if grab == 'tech':
                return match.groups()[0] # first value is the technology
            elif grab == 'width':
                return match.groups()[-1] # last value is the width

def layout_summary(sol, widths, neckin, B):
    remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
    df = pd.DataFrame(sol)
    df = df.fillna(0)
    dff = pd.DataFrame(df.groupby(list(df.columns)).size()).rename(columns={0: 'Doffs'}).reset_index()
    dff.index = dff['Doffs']
    #     dff = dff[dff.columns[:-1]]
    dff[dff.columns] = dff[dff.columns].replace({0:np.nan})
    dff['Loss'] = B - dff[dff.columns[:-1]].sum(axis=1)
    dff = dff.replace(remove_neckin_dic)
    return dff

def highlight_max_row(df):
    return [
        {
            'if': {
                'filter_query': '{{id}} = {}'.format(pd.Series(
                [item for sublist in df.values for item in
                sublist]).unique()[0]),
                'column_id': col
            },
            'backgroundColor': '#3D9970',
            'color': 'white'
        }
        # idxmax(axis=1) finds the max indices of each row
        for (i, col) in enumerate(
            df.columns
        )
    ]

def summarize_results(sol, widths, neckin, B):
    remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
    df = pd.DataFrame(sol)
    df = df.fillna(0)
    dff = pd.DataFrame(df.groupby(list(df.columns)).size()).rename(columns={0: 'freq'}).reset_index()
    dff['Loss'] = B - dff[dff.columns[:-1]].sum(axis=1)
    dff = dff.replace(remove_neckin_dic)
    master = pd.DataFrame()
    for row in dff.index:
        deckle = dff.loc[dff.index == row]
        freq = deckle.freq.values[0]
        loss = deckle.Loss.values[0]
        x = (deckle[deckle.columns[:-2]].values[0]).astype(int)
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        formula = np.vstack((ii,y[ii]))
        read_out = ''
        for i in range(formula.shape[1]-1): #for unique prods
            if formula[0,i] != 0:
                read_out = read_out + ("{}x{} + ".format(formula[0,i], formula[1,i]))
        read_out = read_out + ("{}x{}".format(formula[0,-1], formula[1,-1]))
        current = pd.DataFrame([read_out, freq, loss]).T
        current.columns = ['Formula', 'Doffs', 'Loss']
        master = pd.concat([master, current])
    master = master.sort_values('Doffs', ascending=False).reset_index(drop=True)
    return master

def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles

def data_bars_diverging(df, column, color_above='#3D9970', color_below='#FF4136'):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    col_max = df[column].max()
    col_min = df[column].min()
    ranges = [
        ((col_max - col_min) * i) + col_min
        for i in bounds
    ]
    midpoint = (col_max + col_min) / 2.

    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        min_bound_percentage = bounds[i - 1] * 100
        max_bound_percentage = bounds[i] * 100

        style = {
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'paddingBottom': 2,
            'paddingTop': 2
        }
        if max_bound > midpoint:
            background = (
                """
                    linear-gradient(90deg,
                    white 0%,
                    white 50%,
                    {color_above} 50%,
                    {color_above} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(
                    max_bound_percentage=max_bound_percentage,
                    color_above=color_above
                )
            )
        else:
            background = (
                """
                    linear-gradient(90deg,
                    white 0%,
                    white {min_bound_percentage}%,
                    {color_below} {min_bound_percentage}%,
                    {color_below} 50%,
                    white 50%,
                    white 100%)
                """.format(
                    min_bound_percentage=min_bound_percentage,
                    color_below=color_below
                )
            )
        style['background'] = background
        styles.append(style)

    return styles

def make_layout_registrar(sol, widths, neckin):
    remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
    df = pd.DataFrame(sol)
    df = df.replace(remove_neckin_dic)
    df = df.fillna(0)
    dff = pd.DataFrame(df.groupby(list(df.columns)).size()).rename(columns={0: 'freq'}).reset_index()
    master = pd.DataFrame()
    for row in dff.index:
        deckle = dff.loc[dff.index == row]
        freq = deckle.freq.values[0]
        x = (deckle[deckle.columns[:-1]].values[0]).astype(int)
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        formula = np.vstack((ii,y[ii]))
        read_out = []
        columns = []
        for i in range(formula.shape[1]): #for unique prods
            if formula[0,i] != 0:
                read_out.append(formula[1,i])
                columns.append('{}'.format(formula[0,i]))
        read_out.append(freq)
        columns.append('freq')
        current = pd.DataFrame([read_out])
        current.columns = columns
        master = pd.concat([master, current])
    return master
