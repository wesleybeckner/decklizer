# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from random import shuffle, choice
import copy
import time
from IPython.display import display, clear_output
import dash_table
from dash.dependencies import Input, Output
import urllib
from utils import *
# from genetic import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1',
                  '#FF9DA7', '#9C755F', '#BAB0AC']

#dummy save
# user defined variables
# B = 3850-200 # useful Doff width in MM (RW-Trim)
# L = 22500 # put up, Doff length
# w = [531+8,667+11,574+10] # roll widths with Neck In
# lm = [1035000, 945000, 958188] # material needed in LM

df = pd.read_excel('data/200721_ New Way SCH W3-W6 W14 07.20.20.xlsx',
                   sheet_name='Schedule')
df = df.loc[df['Customer Name'] == 'P & G']
df = df.loc[df['Description'].str.contains('SAM')]
df = df.loc[df['Description'].str.contains('WHITE')] #CYAN, TEAL
df = df.loc[df['CYCLE / BUCKET'] == 'CYCLE 2']
df = df.reset_index(drop=True)
df['Width'] = pd.DataFrame(list(pd.DataFrame(list(df['Description'].str.split(';')))[0].str.split('UN0')))[1]
lm = list(df.groupby('Description')['Total LM Order QTY'].sum().values)
lm = [int(i) for i in lm]
widths = list(df.groupby('Description')['Width'].first().values.astype(int))
B = 4160
L = 17000 # df['LM putup']
neckin = [4, 4, 5, 7, 7, 7] # 158 missing cycle 1, 4 mm knife in
w = list(np.array(widths) + np.array(neckin))
q = [math.ceil(x/L) for x in lm]

s = BinPackingExample(w, q)

schedule_json = df.to_json()



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
sol_json = df.to_json()
df = layout_summary(sol, widths, neckin, B)

stuff = []
for index, width in enumerate(widths):
    stuff.append(
        [
                {
                    'if': {
                        'filter_query': '{{{}}} = {}'.format(col, width),
                        'column_id': str(col)
                    },
                    'backgroundColor': '{}'.format(tableau_colors[index]),
                    'color': 'white'
                }
                for col in range(len(df.columns))
            ]

    )
style_data_conditional = [item for sublist in stuff for item in sublist]

def find_optimum(s,
                 B,
                 widths,
                 neckin,
                 iterations=100,
                 loss_target = 2,
                 max_doff_layouts = 15,
                 max_unique_products = 6,
                 gene_count = 5,
                 doff_min = 1):
    losses = []
    solutions = []
    best_loss = 100
    step = 0
    start_time = time.time()

    while True:
        step += 1
        shuffle(s)
        remain = [B] #initialize list of remaining bin spaces
        sol = [[]]
        binlim = np.inf

        for item in s:#sorted(s, reverse=True): # iter through products
            for j,free in enumerate(remain): #
                if (free >= item) and (len(sol[j]) < binlim): # if theres room
                    remain[j] -= item
                    sol[j].append(item)
                    break
            else: # at this step we need to double the previous (or mult. by min. doff count)
                sol.append([item]) #starts new list in sol list
                remain.append(B-item) #append a new bin and subtract the width of the new item

        genes = []

        df = pd.DataFrame(sol)
        df = df.fillna(0)
        dff = pd.DataFrame(df.groupby(list(df.columns)).size()).\
            rename(columns={0: 'freq'}).reset_index()
        dff = dff[dff.columns[:-1]]
        dff["Loss"] = B - dff.sum(axis=1)
        dff = dff.loc[dff.replace(0, np.nan, inplace=False).nunique(axis=1) <= max_unique_products]
        dff = dff.loc[dff['Loss'] < 100]
        dff = dff.sort_values('Loss').head(gene_count).reset_index(drop=True)

        dff = dff[dff.columns[:-1]]
        dff = dff.fillna(0)

        for row in dff.index:
            gene = list(dff.iloc[row].values)
            gene = [i for i in gene if i != 0]
            genes.append(gene)

        order_remaining = copy.copy(s)
        sol2 = []
        gene_time = time.time() - start_time

        while True:
            backup_order_remaining = copy.copy(order_remaining)
            try:
                new_gene = list(choice(genes))
            except:
                break
            if new_gene in sol2:
                sol2.append(new_gene)
                for item in new_gene:
                    if item in order_remaining:
                        order_remaining.remove(item)
                    elif item == 0:
                        pass
                    else: # new gene over produces item
                        order_remaining = backup_order_remaining
                        sol2.remove(new_gene)
                        genes.remove(new_gene)
                        break
            else:
                check_pass = True
                for mult in range(doff_min):

                    for item in new_gene:
                        if item in order_remaining:
                            order_remaining.remove(item)
                        elif item == 0:
                            pass
                        else: # new gene over produces item

                            check_pass = False
                            break
                    else:

                        continue  # only executed if the inner loop did NOT break
                    if check_pass == False:  # only executed if the inner loop DID break
                        order_remaining = backup_order_remaining
                        genes.remove(new_gene)

                        break
                if check_pass == True:
                    for mult in range(doff_min):
                        sol2.append(new_gene)

        chrome_time = time.time() - gene_time
        #     if step > iterations:
        #         break
        remain2 = [B] #initialize list of remaining bin spaces
        sol3 = [[]]
        binlim = np.inf
        doff_min = 2
        for item in sorted(order_remaining, reverse=True): # iter through products
            for j,free in enumerate(remain2): #
                if (free >= item) and (len(sol3[j]) < binlim): # if theres room,
                    remain2[j] -= item # and we haven't reach bimlim
                    sol3[j].append(item)
                    break
            else: # at this step we need to double the previous (or mult. by min. doff count)
                sol3.append([item]) #starts new list in sol list
                remain2.append(B-item) #append a new bin and subtract the width of the new item
                # subtract
        # loss = sum(remain2) / np.sum(np.sum(sol3)) * 100
        ffd_time = time.time() - chrome_time
        sol_tot = sol2 + sol3
        space_avail = len(sol_tot) * B
        loss = (space_avail - np.sum(np.sum(sol_tot))) / space_avail * 100
        losses.append(loss)
        solutions.append(sol_tot)
        if loss < best_loss:
            best_solution = sol_tot
            best_loss = loss

        if (loss < loss_target) and \
            (summarize_results(sol_tot, widths, neckin, B).shape[0] < 20) and \
            (all(pd.DataFrame(sol_tot).replace(0, np.nan, inplace=False)\
            .nunique(axis=1) <= max_unique_products)):
            break
    return sol_tot, loss

# df = df.rename(columns=columns_dic)
# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options

HIDDEN = html.Div([
    html.Div(id='sol-json',
             style={'display': 'none'},
             children=sol_json),
    html.Div(id='schedule-json',
             style={'display': 'none'},
             children=schedule_json),
    html.Div(id='deckle-schedule-json',
             style={'display': 'none'},
             children=schedule_json),
    html.Div(id='summary-json',
             style={'display': 'none'},
             children=summarize_results(sol, widths, neckin, B).to_json()
             ),
             ])

UPLOAD = html.Div(["Upload Schedule: ",
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '200px',
            'height': '60px',
            # 'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'vertical-align': 'middle',
            'margin': '10px',

            'padding': '5px',
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),],
    # ,className='four columns',
    #     style={
    #     'margin-left': '40px',
    #     },
    id='up-option-1',)

app.layout = html.Div(children=[
    html.H1('Decklizer', style={'display': 'inline-block'}),
    html.Img(src='assets/trump.png', style={'display': 'inline-block',
                                            'height': '50px'}),
    UPLOAD,
    html.Div(["Usable Doff Width (MM): ",
              dcc.Input(id='doff-width', value=B, type='number')]),
    html.Div(["Put Up (MM): ",
              dcc.Input(id='doff-length', value=L, type='number')]),
    html.Div(["Product Widths (MM): ",
              dcc.Input(id='product-width', value=str(widths).split('[')[1].split(']')[0], type='text')]),
    html.Div(["Product Length (LM): ",
              dcc.Input(id='product-length', value=str(lm).split('[')[1].split(']')[0], type='text')]),
    html.Div(["Product Neck In (MM): ",
              dcc.Input(id='neck-in', value=str(neckin).split('[')[1].split(']')[0], type='text')]),
    html.Div(["Max Number of Knives: ",
              dcc.Input(id='max-bins', value='30', type='text')]),
    html.Div(["Max Widths per Doff: ",
              dcc.Input(id='max-widths', value='4', type='text')]),
    html.Div(["Deckle Loss Target (%): ",
              dcc.Input(id='loss-target', value='2', type='text')]),
    html.Br(),
    # html.Div(["EA Iterations: ",
    #           dcc.Input(id='iterations', value=1e3, type='number')]),
    html.Button('Optimize Deckle',
                id='deckle-button',),
    html.Br(),
    html.A('Save Deckle',
                id='save-button',
                download='deckle_pattern.csv',
                href='',
                target='_blank'),
    html.Div([]),
    html.Br(),
    html.Div(["Optimize Schedule For: ",
    dcc.Dropdown(id='optimize-options',
                 multi=False,
                 options=[{'label': i, 'value': i} for i in ['Time (Knife Changes)', 'Late Orders']],
                 placeholder="Select Cloud Dataset",
                 value='Late Orders',
                 className='dcc_control',
                 style={
                        'textAlign': 'center',
                        'width': '200px',
                        'margin': '10px',
                        }
                        ),],
                        # className='four columns',
                        id='optimize-options-div',
                                                    style={
                                                    'margin-right': '40px',
                                                           }
                                                           ),
    html.Button('Create Schedule',
                id='schedule-button',),
    html.Br(),
    html.A('Save Schedule',
                id='save-schedule',
                download='deckle_schedule.csv',
                href='',
                target='_blank'),
    html.Br(),
    html.Br(),
    html.Div(id='my-output'),
    HIDDEN,
    html.Div(id='results',
        children=
        "New Doff Number: {}, Deckle Loss: {:.2f}%".format(len(sol), loss)),
    html.Div(
    children=dash_table.DataTable(id='opportunity-table',
                        sort_action='native',
                        columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                        data = df.to_dict('rows'),
                        style_table={
                            'maxWidth': '1000px',},
                        style_data_conditional=(style_data_conditional
                                                + data_bars(df, 'Doffs')
                                                + data_bars(df, 'Loss'))
                        )),
])

@app.callback(
    Output('save-button', 'href'),
    [Input('summary-json', 'children')])
def update_download_link(sol):
    dff = pd.read_json(sol)
    csv_string = dff.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string

@app.callback(
    Output('save-schedule', 'href'),
    [Input('deckle-schedule-json', 'children')])
def update_download_link(sol):
    dff = pd.read_json(sol)
    csv_string = dff.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string

@app.callback(
    [Output(component_id='results', component_property='children'),
    Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),
    Output('sol-json', 'children'),
    Output('summary-json', 'children'),
    Output('deckle-schedule-json', 'children')],
    [Input(component_id='doff-width', component_property='value'),
    Input(component_id='doff-length', component_property='value'),
    Input(component_id='product-width', component_property='value'),
    Input(component_id='product-length', component_property='value'),
    Input(component_id='neck-in', component_property='value'),
    # Input(component_id='iterations', component_property='value'),
    Input(component_id='max-bins', component_property='value'),
    Input(component_id='max-widths', component_property='value'),
    Input(component_id='loss-target', component_property='value'),
    Input(component_id='optimize-options', component_property='value'),
    Input('deckle-button', 'n_clicks'),
    Input('schedule-json', 'children')
    ]
)
def update_output_div(B, L, wstr, lmstr, neckstr, binlim, widthlim, loss, options,
    button, schedule_json):

    schedule_df = pd.read_json(schedule_json)

    ctx = dash.callback_context
    widthlim = int(widthlim)
    loss = float(loss)

    if (ctx.triggered[0]['prop_id'] == 'deckle-button.n_clicks'):
        widths = []
        for i in wstr.split(','):
            widths.append(int(i))
        neckin = []
        for i in neckstr.split(','):
            neckin.append(int(i))
        lm = []
        for i in lmstr.split(','):
            lm.append(int(i))
        w = list(np.array(widths) + np.array(neckin))

        q = [math.ceil(x/L) for x in lm]
        s = BinPackingExample(w, q)
        B = int(B)
        binlim = int(binlim)

        # sol, remain, loss = simple_genetic(s, B, binlim)
        sol, loss = find_optimum(s, B, widths, neckin,
            max_unique_products=widthlim,
            loss_target=loss)
        if options == 'Late Orders':
            master_schedule = optimize_late_orders(sol, widths, neckin, schedule_df, L)
        for i in sol:
            i.sort()
        sol.sort()

        remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
        df = pd.DataFrame(sol)

        dff = layout_summary(sol, widths, neckin, B)


        return "New Doff Number: {}, Deckle Loss: {:.2f}%".format(len(sol), loss),\
            dff.to_dict('rows'),\
            [{"name": str(i), "id": str(i)} for i in dff.columns],\
            df.to_json(),\
            summarize_results(sol, widths, neckin, B).to_json(),\
            master_schedule.to_json()

if __name__ == '__main__':
    app.run_server(debug=True)
