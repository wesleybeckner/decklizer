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
from dash.dependencies import Input, Output, State
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

df_input_schedule = pd.read_excel('data/200721_ New Way SCH W3-W6 W14 07.20.20.xlsx',
                   sheet_name='Schedule')
df_input_schedule.insert(1, 'Technology', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'tech')))
df_input_schedule.insert(2, 'Color', df_input_schedule['Description'].apply(lambda x: x.split(';')[1] if type(x) == str else None))
df_input_schedule.insert(3, 'Width', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'width')))
df_input_schedule = df_input_schedule[[col for col in df_input_schedule.columns if 'Unnamed' not in str(col)]]
temp = df_input_schedule[['Customer Name', 'Technology', 'Color', 'Width', 'CYCLE / BUCKET',
                          'Description', 'Total LM Order QTY',
                          'Scheduled Ship Date', 'Date order is complete']]

customer = 'P & G'
technology = 'SAM'
color = 'WHITE'
cycle = 'CYCLE 2'

df_filtered = df_input_schedule.loc[df_input_schedule['Customer Name'] == customer]
df_filtered = df_filtered.loc[df_filtered['Description'].str.contains(technology)]
df_filtered = df_filtered.loc[df_filtered['Description'].str.contains(color)] #CYAN, TEAL
df_filtered = df_filtered.loc[df_filtered['CYCLE / BUCKET'] == cycle]
df_filtered.insert(0, 'Block', 1)
df_filtered = df_filtered.reset_index(drop=True)

### if index is broken, create new block
df_filtered = df_filtered.reset_index(drop=True)
df_filtered['Width'] = pd.DataFrame(list(pd.DataFrame(list(df_filtered
    ['Description'].str.split(';')))[0].str.split('UN0')))[1]
lm = list(df_filtered.groupby('Description')['Total LM Order QTY'].sum().values)
lm = [int(i) for i in lm]
widths = list(df_filtered.groupby('Description')['Width'].first().values.astype(int))
doffs_in_jumbo = 6
B = 4160
L = doff_length = 17000 * doffs_in_jumbo
neckin = [4, 4, 5, 7, 7, 7] # 158 missing cycle 1, 4 mm knife in
w = list(np.array(widths) + np.array(neckin))
q = [math.ceil(x/L) for x in lm]
s = BinPackingExample(w, q)

input_schedule_json = df_filtered.to_json()

# pre-calculations
q = [math.ceil(x/L) for x in lm]
s = BinPackingExample(w, q)
sol, remain, loss = FFD(s, B)
df = pd.DataFrame(sol)
sol_json = df.to_json()
sol_df = layout_summary(sol, widths, neckin, B)
sol_df['Doffs'] = sol_df['Doffs']*doffs_in_jumbo


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
                for col in range(len(sol_df.columns))
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
    html.Div(id='layout-sol-json', # the raw solution, initiates from FFD
             style={'display': 'none'},
             children=sol_json),
    html.Div(id='input-schedule-json',
             style={'display': 'none'},
             children=input_schedule_json),
    html.Div(id='input-schedule-processed-json',
             style={'display': 'none'},
             children=input_schedule_json),
    html.Div(id='deckle-schedule-json',
             style={'display': 'none'},
             children=None),
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
            html.A('Select File')
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
    # html.Img(src='assets/trump.png', style={'display': 'inline-block',
    #                                         'height': '50px'}),
    UPLOAD,
    html.Div(
        children=dash_table.DataTable(id='input-schedule-table',
                            sort_action='native',
                            columns=[{"name": str(i), "id": str(i)} for i in temp.columns],
                            data = temp.to_dict('rows'),
                            editable=True,
                            filter_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=True,
                            selected_columns=[],
                            # selected_rows=[0, 1, 2],
                            page_action="native",
                            page_current= 0,
                            style_table={
                                    'maxHeight': '50ex',
                                    'overflowY': 'scroll',
                                    # 'width': '100%',
                                    # 'minWidth': '100%',
                                    'maxWidth': '1000px',
                                }
                            ),
                            ),
    html.Button('Process Bucket',
                id='process-bucket-button',),
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
    # html.Div(["Max Number of Knives: ",
    #           dcc.Input(id='max-bins', value='30', type='text')]),
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
                        columns=[{"name": str(i), "id": str(i)} for i in sol_df.columns],
                        data = sol_df.to_dict('rows'),
                        style_table={
                            'maxWidth': '1000px',},
                        style_data_conditional=(style_data_conditional
                                                + data_bars(sol_df, 'Doffs')
                                                + data_bars(sol_df, 'Loss'))
                        )),
])

### callback to update customer, tech, color, cycle, and block dropdown options
### as well as
### fires anytime value or excel upload is fired
### need to filter the uploaded schedule by the other dropdown values
# @app.callback(
#
# )

### callback to filter uploaded schedule
### fires on button press?

@app.callback(
    [Output('input-schedule-json', 'children'),
     Output('input-schedule-table', 'columns'),
     Output('input-schedule-table', 'data')],
  [Input('upload-data', 'contents'),],
  [State('upload-data', 'filename'),
   State('upload-data', 'last_modified')])
def proccess_upload(contents, filename, date):
    if contents is not None:
        input_schedule_json = parse_contents(contents, filename, date)
        dates = ['Scheduled Ship Date', 'Date order is complete']
        df_input_schedule = pd.read_json(input_schedule_json,
                                convert_dates=dates)
        df_input_schedule.insert(1, 'Technology', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'tech')))
        df_input_schedule.insert(2, 'Color', df_input_schedule['Description'].apply(lambda x: x.split(';')[1] if type(x) == str else None))
        df_input_schedule.insert(3, 'Width', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'width')))
        df_input_schedule = df_input_schedule[[col for col in df_input_schedule.columns if 'Unnamed' not in str(col)]]
        temp = df_input_schedule[['Customer Name', 'Technology', 'Color', 'Width', 'CYCLE / BUCKET',
                                  'Description', 'Total LM Order QTY',
                                  'Scheduled Ship Date', 'Date order is complete']]
        temp['Scheduled Ship Date'] = pd.to_datetime(temp['Scheduled Ship Date'], errors='ignore', unit='ms')
        return [temp.to_json(),
                [{"name": str(i), "id": str(i)} for i in temp.columns],
                temp.to_dict('rows')]


@app.callback(
    [Output('input-schedule-processed-json', 'children'),
    Output(component_id='product-width', component_property='value'),
    Output(component_id='product-length', component_property='value'),
    Output(component_id='neck-in', component_property='value'),],
  [Input('input-schedule-table', 'derived_virtual_selected_rows'),
  Input('input-schedule-table', 'derived_virtual_data'),
  Input('process-bucket-button', 'n_clicks')])
def filter_schedule(rows, data, button):
    ctx = dash.callback_context
    if (ctx.triggered[0]['prop_id'] == 'process-bucket-button.n_clicks'):
        if (data is not None):
            if (len(rows) == 0):
                new_df = pd.DataFrame(data)
            elif (len(rows) > 0):
                new_df = pd.DataFrame(data).iloc[rows]
            widths = list(new_df.groupby('Width')['Total LM Order QTY'].sum()
                    .index.values.astype(int))
            lm = list(new_df.groupby('Width')['Total LM Order QTY'].sum().values)
            neckins = []
            for width in widths:
                if width < 170:
                    neckin = 4
                elif width < 208:
                    neckin = 5
                else:
                    neckin = 7
                neckins.append(neckin)
            if (len(rows) == 0):
                return [pd.DataFrame(data).to_json(),# widths, lm, neckins]
                        str(widths).split('[')[1].split(']')[0],
                        str(lm).split('[')[1].split(']')[0],
                        str(neckins).split('[')[1].split(']')[0]]
            elif (len(rows) > 0):
                return [pd.DataFrame(data).iloc[rows].to_json(),
                        str(widths).split('[')[1].split(']')[0],
                        str(lm).split('[')[1].split(']')[0],
                        str(neckins).split('[')[1].split(']')[0]]

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
    Output('layout-sol-json', 'children'),
    Output('summary-json', 'children'),
    Output('deckle-schedule-json', 'children')],
    [Input(component_id='doff-width', component_property='value'),
    Input(component_id='doff-length', component_property='value'),
    Input(component_id='product-width', component_property='value'),
    Input(component_id='product-length', component_property='value'),
    Input(component_id='neck-in', component_property='value'),
    # Input(component_id='iterations', component_property='value'),
    # Input(component_id='max-bins', component_property='value'),
    Input(component_id='max-widths', component_property='value'),
    Input(component_id='loss-target', component_property='value'),
    Input(component_id='optimize-options', component_property='value'),
    Input('deckle-button', 'n_clicks'),
    Input('input-schedule-processed-json', 'children')
    ]
)
def update_output_div(B, L, wstr, lmstr, neckstr, widthlim, loss, options,
    button, input_schedule_json):

    schedule_df = pd.read_json(input_schedule_json)
    print(schedule_df)

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
            lm.append(int(float(i)))
        w = list(np.array(widths) + np.array(neckin))

        q = [math.ceil(x/L) for x in lm]
        s = BinPackingExample(w, q)
        B = int(B)
        # binlim = int(binlim)

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
        sol_df = pd.DataFrame(sol)

        dff = layout_summary(sol, widths, neckin, B)

        ### replace with doffs_in_jumbo
        dff['Doffs'] = dff['Doffs']*6

        return "New Doff Number: {}, Deckle Loss: {:.2f}%".format(len(sol), loss),\
            dff.to_dict('rows'),\
            [{"name": str(i), "id": str(i)} for i in dff.columns],\
            sol_df.to_json(),\
            summarize_results(sol, widths, neckin, B).to_json(),\
            master_schedule.to_json()

if __name__ == '__main__':
    app.run_server(debug=True)
