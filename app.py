# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from random import shuffle
from IPython.display import display
import dash_table
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#dummy save
# user defined variables
B = 3850-200 # useful Doff width in MM (RW-Trim)
L = 22500 # put up, Doff length
w = [531+8,667+11,574+10] # roll widths with Neck In
lm = [1035000, 945000, 958188] # material needed in LM

def BinPackingExample(w, q):
    """
    returns list, s, of material orders
    of widths w and order numbers q
    """
    s=[]
    for j in range(len(w)):
        for i in range(q[j]):
            s.append(w[j])
    return s

def FFD(s, B):
    """
    first-fit decreasing (FFD) heruistic procedure for finding
    a possibly good upper limit len(s) of the number of bins.
    """
    remain = [B] #initialize list of remaining bin spaces
    sol = [[]]
    for item in sorted(s, reverse=True):
        for j,free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else:
            sol.append([item])
            remain.append(B-item)
    loss = sum(remain) / np.sum(np.sum(sol)) * 100
    return sol, remain, loss

def simple_genetic(s, B, iterations, binlim=np.inf):
    best_loss = 100
    best_sol = [[]]
    best_remain = 0
    loops = 0
    while True:
        loops += 1
        shuffle(s)
        remain = [B] #initialize list of remaining bin spaces
        sol = [[]]
        for item in s:
            for j,free in enumerate(remain):
                if (free >= item) and (len(sol[j]) < binlim):
                    remain[j] -= item
                    sol[j].append(item)
                    break
            else:
                sol.append([item])
                remain.append(B-item)
        loss = sum(remain) / np.sum(np.sum(sol)) * 100
        if loss < best_loss:
            best_loss = loss
            best_sol = sol
            best_remain = remain
        if loops > iterations:
            break
    return best_sol, best_remain, best_loss

# pre-calculations
q = [math.ceil(x/L) for x in lm]
s = BinPackingExample(w, q)
sol, remain, loss = FFD(s, B)
columns_dic= {0: 'first',
1: 'second',
2: 'third',
3: 'fourth',
4:'fifth',
5: 'sixth',
6: 'seventh'}
df = pd.DataFrame(sol)
# df = df.rename(columns=columns_dic)
# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options

app.layout = html.Div(children=[
    html.H1('Decklizer', style={'display': 'inline-block'}),
    html.Img(src='assets/trump.png', style={'display': 'inline-block',
                                            'height': '50px'}),
    html.Div(["Usable Doff Width (M): ",
              dcc.Input(id='doff-width', value=3650, type='number')]),
    html.Div(["Put Up (M): ",
              dcc.Input(id='doff-length', value=22500, type='number')]),
    html.Div(["Product Widths (MM): ",
              dcc.Input(id='product-width', value='531, 667, 574', type='text')]),
    html.Div(["Product Linear Meters: ",
              dcc.Input(id='product-length', value='1035000, 945000, 958188', type='text')]),
    html.Div(["Product Neck In: ",
              dcc.Input(id='neck-in', value='8, 11, 10', type='text')]),
    html.Div(["Max Bin Allocation: ",
              dcc.Input(id='max-bins', value='6', type='text')]),
    html.Br(),
    html.Div(["EA Iterations: ",
              dcc.Input(id='iterations', value=1e3, type='number')]),
    html.Br(),
    html.Button('Opptimize Deckle',
                id='deckle-button',),
    html.Br(),
    html.Br(),
    html.Div(id='my-output'),
    html.Div(id='results',
        children=
        "New Doff Number: {}, Deckle Loss: {:.2f}%".format(len(sol), loss)),
    html.Div(
    children=dash_table.DataTable(id='opportunity-table',
                        columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                        data = df.to_dict('rows'),
                        style_table={
                            'maxWidth': '1000px',},
                        style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{{{}}} = {}'.format(col, pd.Series(
                                [item for sublist in df.values for item in
                                sublist]).unique()[0]),
                                'column_id': str(col)
                            },
                            'backgroundColor': '#FF4136',
                            'color': 'white'
                        }
                        for col in range(10)
                    ] +
                    [
                    {
                        'if': {
                            'filter_query': '{{{}}} = {}'.format(col, pd.Series(
                            [item for sublist in df.values for item in
                            sublist]).unique()[2]),
                            'column_id': str(col)
                        },
                        'backgroundColor': '#3D9970',
                        'color': 'white'
                    }
                    for col in range(10)
                ] +
                [
                {
                    'if': {
                        'filter_query': '{{{}}} = {}'.format(col, pd.Series(
                        [item for sublist in df.values for item in
                        sublist]).unique()[3]),
                        'column_id': str(col)
                    },
                    'backgroundColor': '#0074D9',
                    'color': 'white'
                }
                for col in range(10)
            ] +
            [
            {
                'if': {
                    'filter_query': '{{{}}} = {}'.format(col, pd.Series(
                    [item for sublist in df.values for item in
                    sublist]).unique()[1]),
                    'column_id': str(col)
                },
                'backgroundColor': '#0074D9',
                'color': 'white'
            }
            for col in range(10)
        ]
                        )),
])

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

@app.callback(
    [Output(component_id='results', component_property='children'),
    Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),],
    [Input(component_id='doff-width', component_property='value'),
    Input(component_id='doff-length', component_property='value'),
    Input(component_id='product-width', component_property='value'),
    Input(component_id='product-length', component_property='value'),
    Input(component_id='neck-in', component_property='value'),
    Input(component_id='iterations', component_property='value'),
    Input(component_id='max-bins', component_property='value'),
    Input('deckle-button', 'n_clicks')]
)
def update_output_div(B, L, wstr, lmstr, neckstr, iterations, binlim, button):
    ctx = dash.callback_context

    if (ctx.triggered[0]['prop_id'] == 'deckle-button.n_clicks'):
        w = []
        for i in wstr.split(','):
            w.append(int(i))
        neck = []
        for i in neckstr.split(','):
            neck.append(int(i))
        lm = []
        for i in lmstr.split(','):
            lm.append(int(i))
        w = list(np.array(w) + np.array(neck))

        q = [math.ceil(x/L) for x in lm]
        s = BinPackingExample(w, q)
        B = int(B)
        binlim = int(binlim)

        sol, remain, loss = simple_genetic(s, B, iterations, binlim)
        for i in sol:
            i.sort()
        sol.sort()

        columns_dic= {0: 'first',
        1: 'second',
        2: 'third',
        3: 'fourth',
        4:'fifth',
        5: 'sixth',
        6: 'seventh',
        7: 'eighth',
        8: 'ninth',
        9: 'tenth',
        10: 'eleventh',
        11: 'twelfth'}
        df = pd.DataFrame(sol)
        # df = df.rename(columns=columns_dic)

        return "New Doff Number: {}, Deckle Loss: {:.2f}%".format(len(sol), loss),\
            df.to_dict('rows'),\
            [{"name": str(i), "id": str(i)} for i in df.columns]

if __name__ == '__main__':
    app.run_server(debug=True)
