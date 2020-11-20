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
