# -*- coding: utf-8 -*-
import dash
import dash_auth
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
from engine import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

VALID_USERNAME_PASSWORD_PAIRS = {
    'berrymfg': 'waynesboro'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
server = app.server

tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1',
                  '#FF9DA7', '#9C755F', '#BAB0AC']

setup_df = pd.read_excel('data/200725_WAVA Deckle Optimization Parameters Rev1.xlsx',
                         sheet_name='Deckle Parameters')
speed_df = pd.read_excel('data/200725_WAVA Deckle Optimization Parameters Rev1.xlsx',
                         sheet_name='Product Parameters')

setup_json = setup_df.to_json()
speed_json = speed_df.to_json()

df_input_schedule = pd.read_excel('data/200721_ New Way SCH W3-W6 W14 07.20.20.xlsx',
                   sheet_name='Schedule')
df_input_schedule.insert(1, 'Technology', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'tech')))
df_input_schedule.insert(2, 'Color', df_input_schedule['Description'].apply(lambda x: x.split(';')[1] if type(x) == str else None))
df_input_schedule.insert(3, 'Width', df_input_schedule['Description'].apply(lambda x: parse_description(x, 'width')))
df_input_schedule = df_input_schedule[[col for col in df_input_schedule.columns if 'Unnamed' not in str(col)]]
df_input_schedule['Total LM Order QTY'] = df_input_schedule['Total LM Order QTY'].round(1)
temp = df_input_schedule[['Customer Name', 'Technology', 'Color', 'Width', 'CYCLE / BUCKET',
                          'Description', 'Total LM Order QTY', 'LM putup',
                          'Scheduled Ship Date', 'Date order is complete']]
temp['Scheduled Ship Date'] = pd.to_datetime(temp['Scheduled Ship Date'], errors='coerce')
temp["Scheduled Ship Date"] = temp["Scheduled Ship Date"].dt.strftime("%Y-%m-%d")
temp["Date order is complete"] = temp["Date order is complete"].dt.round('T')

customer = 'P & G'
technology = 'SAM'
color = 'WHITE'
cycle = 'CYCLE 2'

schedule_df = df_input_schedule.loc[df_input_schedule['Customer Name'] == customer]
schedule_df = schedule_df.loc[schedule_df['Description'].str.contains(technology)]
schedule_df = schedule_df.loc[schedule_df['Description'].str.contains(color)] #CYAN, TEAL
schedule_df = schedule_df.loc[schedule_df['CYCLE / BUCKET'] == cycle]
schedule_df = schedule_df.loc[schedule_df['Total LM Order QTY'] > 0]
schedule_df.insert(0, 'Block', 1)
schedule_df = schedule_df.reset_index(drop=True)
schedule_df['Order Number'] = schedule_df.index + 1

### if index is broken, create new block
schedule_df = schedule_df.reset_index(drop=True)
schedule_df['Width'] = pd.DataFrame(list(pd.DataFrame(list(schedule_df
    ['Description'].str.split(';')))[0].str.split('UN0')))[1]
lm = list(schedule_df.groupby('Description')['Total LM Order QTY'].sum().values)
lm = [int(i) for i in lm]
widths = list(schedule_df.groupby('Description')['Width'].first().values.astype(int))
doffs_in_jumbo = 6
B = 4160
doff_length = 17000
L = 17000 * doffs_in_jumbo
neckin = [4, 4, 5, 7, 7, 7] # 158 missing cycle 1, 4 mm knife in
w = list(np.array(widths) + np.array(neckin))
q = [math.ceil(x/L) for x in lm]
s = BinPackingExample(w, q)
start_date_time = schedule_df['Date order is complete'][0]
input_schedule_json = schedule_df.to_json()

################################################################################
################ pre-calculations to populate tables on load ###################
################################################################################
q = [math.ceil(x/L) for x in lm]
s = BinPackingExample(w, q)
sol, remain, loss = FFD(s, B)
df = pd.DataFrame(sol)
sol_json = df.to_json()
sol_df = layout_summary(sol, widths, neckin, B)
sol_df['Doffs'] = sol_df['Doffs']*doffs_in_jumbo
master_schedule, extras = optimize_schedule(sol, widths, neckin,
    schedule_df, L, setup_df, speed_df, doffs_in_jumbo, start_date_time, "Time (Knife Changes)")
master_schedule['Scheduled Ship Date'] = pd.to_datetime(master_schedule['Scheduled Ship Date'], errors='coerce')
master_schedule["Scheduled Ship Date"] = master_schedule["Scheduled Ship Date"].dt.strftime("%Y-%m-%d")

master_schedule["Completion Date"] = master_schedule["Completion Date"].dt.round('T')

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
deckle_time = master_schedule.iloc[-1]['Completion Date'] - \
                master_schedule.iloc[0]['Completion Date']
deckle_time = ("{}".format(deckle_time)).split(".")[0]
################################################################################
################ pre-calculations to populate tables on load ###################
################################################################################


# df = df.rename(columns=columns_dic)
# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options

HIDDEN = html.Div([
    html.Div(id='layout-sol-json', # the raw solution, initiates from FFD
             style={'display': 'none'},
             children=sol_json),
    html.Div(id='speed-json', # the raw solution, initiates from FFD
             style={'display': 'none'},
             children=speed_json),
    html.Div(id='setup-json', # the raw solution, initiates from FFD
             style={'display': 'none'},
             children=setup_json),
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
                            export_format='xlsx',
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
    html.Br(),
    html.Div(["Usable Doff Width (MM): ",
              dcc.Input(id='doff-width', value=B, type='number')]),
    html.Div(["Put Up (LM): ",
              dcc.Input(id='doff-length', value=doff_length, type='number')]),
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
    html.Div(["Max Layouts: ",
              dcc.Input(id='max-layouts', value=20, type='number')]),
    html.Div(["Deckle Loss Target (%): ",
              dcc.Input(id='loss-target', value='2', type='text')]),
    html.Div(["Doffs per Jumbo: ",
              dcc.Input(id='doffs-per-jumbo', value='6', type='text')]),
    html.Br(),
    # html.Div(["EA Iterations: ",
    #           dcc.Input(id='iterations', value=1e3, type='number')]),
    html.Button('Optimize Deckle',
                id='deckle-button',),
    html.Br(),
    html.Br(),
    HIDDEN,
    html.Div(id='results',
        children=
        "Deckle Loss: {:.2f}%".format(loss)),
    html.Br(),
    html.Div(
    children=dash_table.DataTable(id='opportunity-table',
                        sort_action='native',
                        columns=[{"name": str(i), "id": str(i)} for i in sol_df.columns],
                        data = sol_df.to_dict('rows'),
                        export_format='xlsx',
                        style_table={
                            'maxWidth': '1000px',},
                        style_data_conditional=(style_data_conditional
                                                + data_bars(sol_df, 'Doffs')
                                                + data_bars(sol_df, 'Loss'))
                        )),
    html.Br(),
    html.Br(),
    html.Div(["Deckle Start Date: ",
    dcc.DatePickerSingle(
        id='deckle-date',
        min_date_allowed=datetime.datetime(2020, 8, 5),
        max_date_allowed=datetime.datetime(2021, 9, 19),
        initial_visible_month=start_date_time,
        date=str(start_date_time),
    ),]),
    html.Br(),
    html.Div(["Optimize Schedule For: ",
    dcc.Dropdown(id='optimize-deckle-schedule-options',
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
                        id='optimize-deckle-schedule-options-div',
                                                    style={
                                                    'margin-right': '40px',
                                                           }
                                                           ),
    html.Button('Create Schedule',
                id='schedule-button',),
    html.Br(),
    html.Br(),
    html.Div(id='schedule-results',
        children=
        "Total Deckle Time: {}".format(deckle_time)),
    html.Br(),
    html.Div(
    children=dash_table.DataTable(id='deckle-schedule-table',
                        sort_action='native',
                        columns=[{"name": str(i), "id": str(i)} for i in master_schedule.columns],
                        data = master_schedule.to_dict('rows'),
                        export_format='xlsx',
                        style_table={
                                'maxHeight': '50ex',
                                'overflowY': 'scroll',
                                # 'width': '100%',
                                # 'minWidth': '100%',
                                'maxWidth': '1000px',
                            },
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
        df_input_schedule['Total LM Order QTY'] = df_input_schedule['Total LM Order QTY'].round(1)
        temp = df_input_schedule[['Customer Name', 'Technology', 'Color', 'Width', 'CYCLE / BUCKET',
                                  'Description', 'Total LM Order QTY',
                                  'Scheduled Ship Date', 'Date order is complete']]
        temp['Scheduled Ship Date'] = pd.to_datetime(temp['Scheduled Ship Date'], errors='coerce', unit='ms')
        temp["Scheduled Ship Date"] = temp["Scheduled Ship Date"].dt.strftime("%Y-%m-%d")
        return [temp.to_json(),
                [{"name": str(i), "id": str(i)} for i in temp.columns],
                temp.to_dict('rows')]


@app.callback(
    [Output('input-schedule-processed-json', 'children'),
    Output(component_id='product-width', component_property='value'),
    Output(component_id='product-length', component_property='value'),
    Output(component_id='neck-in', component_property='value'),
    Output(component_id='deckle-date', component_property='date'),
    Output(component_id='doff-length', component_property='value')],
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
            new_df = new_df.loc[new_df['Total LM Order QTY'] > 0]
            widths = list(new_df.groupby('Width')['Total LM Order QTY'].sum()
                    .index.values.astype(int))
            lm = [round(i) for i in list(new_df.groupby('Width')['Total LM Order QTY'].sum().values)]
            neckins = []
            doff_length = new_df['LM putup'].values[0]
            for width in widths:
                if width < 170:
                    neckin = 4
                elif width < 208:
                    neckin = 5
                else:
                    neckin = 7
                neckins.append(neckin)
            start_date_time = new_df['Date order is complete'][0]
            if (len(rows) == 0):
                return [pd.DataFrame(data).to_json(),# widths, lm, neckins]
                        str(widths).split('[')[1].split(']')[0],
                        str(lm).split('[')[1].split(']')[0],
                        str(neckins).split('[')[1].split(']')[0],
                        str(start_date_time),
                        doff_length]
            elif (len(rows) > 0):
                return [pd.DataFrame(data).iloc[rows].to_json(),
                        str(widths).split('[')[1].split(']')[0],
                        str(lm).split('[')[1].split(']')[0],
                        str(neckins).split('[')[1].split(']')[0],
                        str(start_date_time),
                        doff_length]

@app.callback(
    [Output(component_id='results', component_property='children'),
    Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),
    Output('layout-sol-json', 'children'),
    Output('summary-json', 'children'),
    Output('opportunity-table', 'style_data_conditional'),
    ],
    [Input(component_id='doff-width', component_property='value'),
    Input(component_id='doff-length', component_property='value'),
    Input(component_id='product-width', component_property='value'),
    Input(component_id='product-length', component_property='value'),
    Input(component_id='neck-in', component_property='value'),
    Input(component_id='max-widths', component_property='value'),
    Input(component_id='loss-target', component_property='value'),
    Input('deckle-button', 'n_clicks'),
    Input('input-schedule-processed-json', 'children'),
    Input('setup-json', 'children'),
    Input('speed-json', 'children'),
    Input('doffs-per-jumbo', 'value'),
    Input('max-layouts', 'value'),
    ]
)
def update_output_div(B, L, wstr, lmstr, neckstr, widthlim, loss,# options,
    button, input_schedule_json, setup_json, speed_json, doffs_in_jumbo,
    max_doff_layouts, DEBUG=False):
    setup_df = pd.read_json(setup_json)
    speed_df = pd.read_json(speed_json)
    doffs_in_jumbo = int(doffs_in_jumbo)
    L = L * doffs_in_jumbo



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
        if DEBUG:
            print(widths)

        # sol, remain, loss = simple_genetic(s, B, binlim)
        sol, loss = find_optimum(s, B, widths, neckin,
            max_unique_products=widthlim,
            loss_target=loss, max_doff_layouts=max_doff_layouts)
        for i in sol:
            i.sort()
        sol.sort()

        remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
        sol_df = pd.DataFrame(sol)

        dff = layout_summary(sol, widths, neckin, B)

        ### replace with doffs_in_jumbo
        dff['Doffs'] = dff['Doffs']*doffs_in_jumbo

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
                        for col in range(len(dff.columns))
                    ]

            )
        style_data_conditional = [item for sublist in stuff for item in sublist]

        return "Deckle Loss: {:.2f}%".format(loss),\
            dff.to_dict('rows'),\
            [{"name": str(i), "id": str(i)} for i in dff.columns],\
            sol_df.to_json(),\
            summarize_results(sol, widths, neckin, B).to_json(),\
            (style_data_conditional
                                    + data_bars(dff, 'Doffs')
                                    + data_bars(dff, 'Loss'))
            # master_schedule.to_json(date_unit='ns')

@app.callback(
    [Output('deckle-schedule-json', 'children'),
    Output('deckle-schedule-table', 'data'),
    Output('deckle-schedule-table', 'columns'),
    Output('schedule-results', 'children')],
    [Input(component_id='doff-width', component_property='value'),
    Input(component_id='doff-length', component_property='value'),
    Input(component_id='product-width', component_property='value'),
    Input(component_id='product-length', component_property='value'),
    Input(component_id='neck-in', component_property='value'),
    # Input(component_id='iterations', component_property='value'),
    # Input(component_id='max-bins', component_property='value'),
    Input(component_id='max-widths', component_property='value'),
    Input(component_id='loss-target', component_property='value'),
    Input('schedule-button', 'n_clicks'),
    Input('input-schedule-processed-json', 'children'),
    Input('setup-json', 'children'),
    Input('speed-json', 'children'),
    Input('doffs-per-jumbo', 'value'),
    Input('layout-sol-json', 'children'),
    Input('deckle-date', 'date'),
    Input(component_id='optimize-deckle-schedule-options', component_property='value'),
    ]
)
def update_output_div(B, L, wstr, lmstr, neckstr, widthlim, loss, #options,
    button, input_schedule_json, setup_json, speed_json, doffs_in_jumbo,
    sol_json, start, objective, DEBUG=True):
    if 'T0' in start:
        start_date = datetime.datetime.strptime(start.split('.')[0], '%Y-%m-%dT0%H:%M:%S')
    elif 'T1' in start:
        start_date = datetime.datetime.strptime(start.split('.')[0], '%Y-%m-%dT1%H:%M:%S')
    elif len(start.split(' ')) == 1:
        start_date = datetime.datetime.strptime(start.split('.')[0], '%Y-%m-%d')
    else:
        start_date = datetime.datetime.strptime(start.split('.')[0], '%Y-%m-%d %H:%M:%S')

    sol_df = pd.read_json(sol_json)
    sol = sol_df.values.tolist()
    sol = [[j for j in i if j > 0] for i in sol]
    setup_df = pd.read_json(setup_json)
    speed_df = pd.read_json(speed_json)
    doffs_in_jumbo = int(doffs_in_jumbo)
    L = L * doffs_in_jumbo

    dates = ['Scheduled Ship Date', 'Date order is complete']
    # df_input_schedule = pd.read_json(input_schedule_json,
    #                         convert_dates=dates)
    schedule_df = pd.read_json(input_schedule_json,
                            convert_dates=dates)
    schedule_df = schedule_df.loc[schedule_df['Total LM Order QTY'] > 0]

    ctx = dash.callback_context
    widthlim = int(widthlim)
    loss = float(loss)

    if (ctx.triggered[0]['prop_id'] == 'schedule-button.n_clicks'):
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
        if DEBUG:
            print(widths)

        # sol, remain, loss = simple_genetic(s, B, binlim)
        # sol, loss = find_optimum(s, B, widths, neckin,
        #     max_unique_products=widthlim,
        #     loss_target=loss)
        # if options == 'Late Orders':
            # master_schedule = optimize_late_orders(sol, widths, neckin, schedule_df, L)
        print(objective)
        master_schedule, extras = optimize_schedule(sol, widths, neckin,
            schedule_df, L, setup_df, speed_df, doffs_in_jumbo, start_date,
            objective)

        ### should not create inventory at this point,
        ### but we don't know how jumbos are divvied between orders
        out = "Inventory created: "
        for col in extras:
            if extras[col][0] != 0:
                out += "{} x {:.0f}; ".format(col, extras[col][0])
        out = out[:-2]

        deckle_time = master_schedule.iloc[-1]['Completion Date']\
                        - master_schedule.iloc[0]['Completion Date']
        deckle_time = ("{}".format(deckle_time)).split(".")[0]

        master_schedule['Scheduled Ship Date'] = pd.to_datetime(master_schedule['Scheduled Ship Date'], errors='coerce')
        master_schedule["Scheduled Ship Date"] = master_schedule["Scheduled Ship Date"].dt.strftime("%Y-%m-%d")
        master_schedule["Completion Date"] = master_schedule["Completion Date"].dt.round('T')
        # for i in sol:
        #     i.sort()
        # sol.sort()

        # remove_neckin_dic = {i+j: i for i, j in zip(widths,neckin)}
        # sol_df = pd.DataFrame(sol)
        #
        # dff = layout_summary(sol, widths, neckin, B)

        ### replace with doffs_in_jumbo
        # dff['Doffs'] = dff['Doffs']*doffs_in_jumbo
        pd.options.display.max_columns = 999
        print(master_schedule.head())
        return [master_schedule.to_json(date_unit='ns'),
        master_schedule.to_dict('rows'),
        [{"name": str(i), "id": str(i)} for i in master_schedule.columns],
        "Total Deckle Time: {}".format(deckle_time)]

if __name__ == '__main__':
    app.run_server(debug=True)
