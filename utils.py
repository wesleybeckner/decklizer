import numpy as np
import pandas as pd
import math
import urllib
import copy
import base64
import dash_html_components as html
import io
import re
import datetime
import time
from random import shuffle, choice
from IPython.display import display, clear_output
import urllib


def find_optimum(s,
                 B,
                 widths,
                 neckin,
                 iterations=100,
                 loss_target = 2,
                 max_doff_layouts = 20,
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
            print(best_loss)

        if (loss < loss_target) and \
            (summarize_results(sol_tot, widths, neckin, B).shape[0] <
            (max_doff_layouts+1)) and \
            (all(pd.DataFrame(sol_tot).replace(0, np.nan, inplace=False)\
            .nunique(axis=1) <= max_unique_products)):
            break
    return sol_tot, loss


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
        return html.Div([
            'There was an error processing this file.'
        ])
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

def genetic(s, B, iterations=100):
    remain = [B] #initialize list of remaining bin spaces
    sol = [[]]
    binlim = np.inf
    doff_min = 2
    for item in sorted(s, reverse=True): # iter through products
        for j,free in enumerate(remain): #
            if (free >= item) and (len(sol[j]) < binlim): # if theres room, and we haven't reach bimlim
                remain[j] -= item
                sol[j].append(item)
                break
        else: # at this step we need to double the previous (or mult. by min. doff count)
            sol.append([item]) #starts new list in sol list
            remain.append(B-item) #append a new bin and subtract the width of the new item


    genes = []
    df = pd.DataFrame(sol)
    dff = pd.DataFrame(df.groupby(list(df.columns)).size()).rename(columns={0: 'freq'}).reset_index()
    dff = dff[dff.columns[:-1]]
    for row in dff.index:
        genes.append(list(dff.iloc[row].values))

    order_remaining = copy.copy(s)
    sol2 = []
    iterations = 100
    step = 0
    while True:
        step += 1
        backup_order_remaining = copy.copy(order_remaining)
        try:
            new_gene = list(choice(genes))
        except:
            break
        sol2.append(new_gene)
        for item in new_gene:
            if item in order_remaining:
                order_remaining.remove(item)
            else: # new gene over produces item
                order_remaining = backup_order_remaining
                sol2.remove(new_gene)
                genes.remove(new_gene)
                break

    #     if step > iterations:
    #         break
    remain = [B] #initialize list of remaining bin spaces
    sol3 = [[]]
    binlim = np.inf
    doff_min = 2
    for item in sorted(order_remaining, reverse=True): # iter through products
        for j,free in enumerate(remain): #
            if (free >= item) and (len(sol3[j]) < binlim): # if theres room, and we haven't reach bimlim
                remain[j] -= item
                sol3[j].append(item)
                break
        else: # at this step we need to double the previous (or mult. by min. doff count)
            sol3.append([item]) #starts new list in sol list
            remain.append(B-item) #append a new bin and subtract the width of the new item
            # subtract
    # loss = sum(remain) / np.sum(np.sum(sol3)) * 100
    sol_tot = sol2 + sol3
    space_avail = len(sol_tot) * B
    loss = (space_avail - np.sum(np.sum(sol_tot))) / space_avail * 100
    return sol_tot, loss

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

def optimize_late_orders(sol, widths, neckin, df, L, DEBUG=False):
    """
    Parameters
    ----------
    sol: list
        optimized layouts
    widths: list
        list of product widths
    neckin: list
        list of product neckins
    df: DataFrame
        pandas DataFrame of schedule
    L: int
        put up

    """
    extras = pd.DataFrame(np.zeros(len(widths))).T
    master2 = make_layout_registrar(sol, widths, neckin)
    master2.columns=master2.columns.str.strip()
    extras.columns = master2.columns[:-1]
    schedule = []
    layout_pattern = 0
    old_width = width = None
    for row1 in df.index:
        current = df.iloc[row1][['Total LM Order QTY', 'Width', 'Scheduled Ship Date']]
        doffs = math.ceil(current['Total LM Order QTY'] / L) # QTY
        width = current['Width']
        if DEBUG:
            print(width)
        width = str(width)
        if width == old_width:
            layout_pattern -= 1
        if DEBUG:
            print(master2.head())
        master2 = master2.sort_values(width, ascending=False)
        master2 = master2.reset_index(drop=True)
        target_doffs = extras.iloc[0][width]
        if DEBUG:
            print(doffs, width, target_doffs)
        for row in master2.index:
            layout_pattern += 1
            for count in range(master2.iloc[row]['freq'].astype(int)):
                target_doffs += master2.iloc[row][width]
                master2.at[row, 'layout number'] = layout_pattern
                schedule.append(master2.iloc[row])
                master2.at[row, 'freq'] = master2.at[row, 'freq'] - 1

                for master_index in master2.iloc[row].index[:-2]:
                    if (math.isnan(master2.iloc[row][master_index]) == False):
                        extras.iloc[0][master_index] = extras.iloc[0][master_index] + master2.iloc[row][master_index]
                if target_doffs > doffs:
                    break
            else:
                continue
            break
        old_width = width
        extra = target_doffs - doffs
        extras.iloc[0][width] = extra
        if DEBUG:
            print(extras)

    master_schedule = pd.DataFrame()
    sorted_schedule = pd.DataFrame(schedule)
    dff = pd.DataFrame(sorted_schedule.groupby('layout number').size()).rename(columns={0: 'Doffs'}).reset_index()
    for layout_number in dff['layout number']:
        deckle = sorted_schedule.loc[sorted_schedule['layout number'] == layout_number]
        formula = deckle.reset_index(drop=True).iloc[0].dropna()
        freq = dff.loc[dff['layout number'] == layout_number]['Doffs'].values[0]
        read_out = ''
        for i in range(formula.shape[0]-3): #for unique prods
            read_out = read_out + ("{}x{} + ".format(formula.index[i], formula.iloc[i].astype(int)))
        read_out = read_out + ("{}x{}".format(formula.index[-3], formula.iloc[-3].astype(int)))
        current = pd.DataFrame([read_out, int(freq)]).T
        current.columns = ['Formula', 'Doffs']
        master_schedule = pd.concat([master_schedule, current])
    master_schedule = master_schedule.reset_index(drop=True)
    return master_schedule

def optimize_schedule(sol, widths, neckin, df_filtered, L, setup_df, speed_df,
                      doffs_in_jumbo, start_date_time, objective='knife changes', DEBUG=False):

    #####################################################
    # Step 1: Create Schedule According to Order Sequence
    #####################################################
    extras = pd.DataFrame(np.zeros(len(widths))).T
    master2 = make_layout_registrar(sol, widths, neckin)
    master2.columns = master2.columns.str.strip()
    if DEBUG:
        print(master2.head())
    extras.columns = master2.columns[:-1]
    schedule = []
    schedule_with_order_info = []
    layout_pattern = 0
    old_width = width = None
    completed_orders = []
    df_filtered['Order Number'] = df_filtered.index + 1

    if objective == 'Late Orders':
        #### logic for jumbo rolls should go in first loop
        for row1 in df_filtered.index:
            clear_output(wait=True)
            ### select schedule data
            current_scheduled = df_filtered.iloc[row1][['Total LM Order QTY', 'Width', 'Scheduled Ship Date']]
            current_scheduled['Order Number'] = row1+1
            ship_date = current_scheduled['Scheduled Ship Date']
            doffs = math.ceil(current_scheduled['Total LM Order QTY'] / L) # QTY
            width = str(current_scheduled['Width'])
            if DEBUG:
                print(width)

            ### check if we've entered a new product
            ### no setup changes here, we need to see if the layout has changed
            if width == old_width:
                layout_pattern -= 1

            ### sort by the current scheduled width in master 2
            master2 = master2.sort_values(str(width), ascending=False)
            master2 = master2.reset_index(drop=True)
            if DEBUG:
                print(master2.head())

            ### calc how many new doffs need to be made from inventory
            target_doffs = extras.iloc[0][width]
            if DEBUG:
                print(pd.DataFrame(data=[[doffs, width, target_doffs]],
                                   columns=['required doffs', 'width', 'doffs made']))
                print(doffs, width, target_doffs)

            ### only proceed if doffs are available in layout registrar
            ### and not in inventory
            if ((any(master2.loc[master2[width] > 0]['freq'] > 0))
                and (target_doffs <= doffs)):

            ### go through the rows in the master2 registrar and check for
            ### layout patterns that contain our current width

            ### Every row is a new layout unless the outer
            ### loop has incremented, then only maybe is it
            ### a new layout
                for row in master2.index:

                    # if none of the width are in the layout then break out
                    if np.isnan(master2.iloc[row][width]):
                        break
                    layout_pattern += 1

                    # looping through available doffs in the layout
                    for count in range(master2.iloc[row]['freq'].astype(int)):

                        # add to target doffs
                        target_doffs += master2.iloc[row][width]

                        # add to number of times we've made this layout
                        master2.at[row, 'layout number'] = layout_pattern


                        new_layout_and_scheduled_product = pd.concat([master2.iloc[row],
                                                                      current_scheduled])
                        schedule_with_order_info.append(new_layout_and_scheduled_product)
                        schedule.append(master2.iloc[row])
                        master2.at[row, 'freq'] = master2.at[row, 'freq'] - 1

                        ### Tabluate the other widths that were made on the layout
                        for master_index in master2.iloc[row].index[:-2]:
                            if (math.isnan(master2.iloc[row][master_index]) == False):
                                ### tabulate extras for other widths in the layout
                                extras.iloc[0][master_index] = extras.iloc[0][master_index] +\
                                    master2.iloc[row][master_index]
                                completed_orders.append
                        if target_doffs >= doffs:

                            ### add to registrar of completed orders.
                            completed_orders.append(current_scheduled['Order Number'])

                            ### now that we've finished this order, we want to see if we've completed any other orders as well


                            break
                    ### for exiting nested for loop
                    else:
                        continue
                    break
            old_width = width
            extra = target_doffs - doffs
            extras.iloc[0][width] = extra
            if DEBUG:
                print(extras)

        sorted_schedule_with_order_info = pd.DataFrame(schedule_with_order_info)
        sorted_schedule_with_order_info = sorted_schedule_with_order_info.reset_index(drop=True)
        sorted_schedule = pd.DataFrame(schedule)
        sorted_schedule = sorted_schedule.reset_index(drop=True)
    else:
        # when we sort master, this will be similar to how we sort
        # according to the desirable width in the optimize for late
        # order algorithm. We will however do this for every width

        for width in widths:
            master2 = master2.sort_values(str(width), ascending=False)
            master2 = master2.reset_index(drop=True)

        # now we need to go through the schedule and match layouts
        # with orders

        # go through the layouts
        for row in master2.index:

            # go through doffs in the layout
            for count in range(master2.iloc[row]['freq'].astype(int)):

                # here we will grow our extras dataframe. We will use this
                # to check against orders that are fullfilled.

                ### Tabluate the other widths that were made on the layout
                for master_index in master2.iloc[row].index[:-1]: # no layout number col, -2 in other algorithm

                    if (math.isnan(master2.iloc[row][master_index]) == False):

                        ### tabulate extras for other widths in the layout
                        extras.iloc[0][master_index] = extras.iloc[0][master_index] +\
                            master2.iloc[row][master_index]

                # go through the orders
                added_this_roll = []
                if df_filtered.shape[0] > 0:
                    for row1 in df_filtered.index:
                        current_scheduled = df_filtered.iloc[row1][['Total LM Order QTY', 'Width',
                                                                         'Scheduled Ship Date',
                                                                         'Order Number']]
                        order_number = current_scheduled['Order Number']
                        ship_date = current_scheduled['Scheduled Ship Date']
                        doffs = math.ceil(current_scheduled['Total LM Order QTY'] / L) # QTY
                        width = str(current_scheduled['Width'])
                        if extras[width][0] > doffs:

                            completed_orders.append(order_number)
                            extra = extras[width][0] - doffs
                            extras.iloc[0][width] = extra
                            new_layout_and_scheduled_product = pd.concat([master2.iloc[row],
                                                                          current_scheduled])
                            schedule_with_order_info.append(new_layout_and_scheduled_product)
                            added_this_roll.append(order_number)
                # if no order completed just append the layout
                if len(added_this_roll) == 0:
                    schedule_with_order_info.append(master2.iloc[row])
                # if two orders completed append both
                elif len(added_this_roll) == 2:
                    schedule_with_order_info[-1][-1] =  "{} and {}".format(added_this_roll[0], added_this_roll[1])
                    schedule_with_order_info.pop(-2)
                elif len(added_this_roll) > 2:
                    print('this is not tested')
                    schedule_with_order_info[-1][-1] = str(added_this_roll).split('[')[-1].split(']')[0]
                    # schedule_with_order_info.pop(-len(added_this_roll))
                df_filtered = df_filtered[~df_filtered['Order Number'].isin(completed_orders)]
                df_filtered = df_filtered.reset_index(drop=True)
                master2.loc[0, 'freq'] -= 1


        sorted_schedule_with_order_info = pd.DataFrame(schedule_with_order_info)
        sorted_schedule_with_order_info = sorted_schedule_with_order_info.reset_index(drop=True)
        sorted_schedule_with_order_info[['Total LM Order QTY',
       'Width', 'Scheduled Ship Date', 'Order Number']] = \
        sorted_schedule_with_order_info[['Total LM Order QTY',
       'Width', 'Scheduled Ship Date', 'Order Number']].fillna(method='bfill')

    #####################################################
    # Step 2: Add Times Based on Rates/Changeovers
    #####################################################
    ### requires setup_df, speed_df

    # slitter_speed = speed_df.loc[(speed_df['Customer Name'] == customer) &
    #          (speed_df['Description'].str.contains(technology)) &
    #          (speed_df['Description'].str.contains(color)), 'Slittter Speed (m/min)'].reset_index(drop=True)[0]
    slitter_speed = 1300

    # add column 'completion date/time'
    # for row in schedule, calculate the completion time
    sorted_schedule_with_order_info['Completion Date'] = None

    ### choose starte date/time
    # start_date_time = datetime.datetime(2020, 8, 12)
    if DEBUG:
        print("start time: {}".format(start_date_time))

    ### set changeover times
    jumbo_change = setup_df.loc[setup_df['Slitter Set-up'] ==
                                'Jumbo roll only']['Time (minutes)'].values[0]
    jumbo_change = datetime.timedelta(minutes=jumbo_change)
    if DEBUG:
        print("jumbo change: {}".format(jumbo_change))
    jumbo_and_knife_change = setup_df.loc[setup_df['Slitter Set-up'] ==
                                          'AMB & Arium']['Time (minutes)'].values[0]
    jumbo_and_knife_change = datetime.timedelta(minutes=jumbo_and_knife_change)
    if DEBUG:
        print("jumbo and knife change: {}".format(jumbo_and_knife_change))

    ### print slitter speed and doff length
    if DEBUG:
        print("slitter speed (m/min): {}".format(slitter_speed))
        print("doff length (m): {}".format(L))
    prior_completion_date_time = start_date_time
    match = '\d\d\d'
    layout_columns = [i for i in sorted_schedule_with_order_info.columns if re.match(match, i)]

    ### zero-out the 'prior layout' so algorithm knows jumbo + knife change
    prior_layout = sorted_schedule_with_order_info.iloc[0][layout_columns]
    for col in prior_layout.index:
        prior_layout[col] = 0
    layout_number = 0

    schedule_with_change_over = pd.DataFrame()
    ### transition times are for loading the current jumbo
    for row in sorted_schedule_with_order_info.index:

        layout = sorted_schedule_with_order_info.iloc[row][layout_columns]
        if all(prior_layout.fillna(0) == layout.fillna(0)):
            transition_time = jumbo_change
        else:
            transition_time = jumbo_and_knife_change
            layout_number += 1
        if DEBUG:
            print("transition time: {}".format(transition_time))
        run_time = L / slitter_speed
        run_time = datetime.timedelta(minutes=run_time)
        if DEBUG:
            print("run time: {}".format(run_time))
        completion_time = transition_time + run_time
        completion_date_time = prior_completion_date_time + completion_time
        if DEBUG:
            print("completion date/time: {}".format(completion_date_time))

        ### change main df
        sorted_schedule_with_order_info['Completion Date'][row] = completion_date_time
        # sorted_schedule_with_order_info['layout number'][row] = layout_number

        ### with changeover rows
        current_changeover = pd.DataFrame(sorted_schedule_with_order_info.columns)
        current_changeover = current_changeover.set_index(0)
        current_changeover = current_changeover.T
        current_changeover = current_changeover.append({'Order Number': 'Changeover'}, ignore_index=True)
        current_changeover['Completion Date'] = prior_completion_date_time + transition_time
        current_order = pd.DataFrame(sorted_schedule_with_order_info.iloc[row]).T
        current_combined = current_changeover.append(current_order, sort=False, ignore_index=True)

        schedule_with_change_over = pd.concat([schedule_with_change_over, current_combined])

        prior_completion_date_time = completion_date_time
        prior_layout = layout
        clear_output(wait=True)

    schedule_with_change_over = schedule_with_change_over.reset_index(drop=True)

    #####################################################
    # Step 3: Create Summary With Changeover Rows
    #####################################################

    master_schedule = pd.DataFrame()
    unformatted_schedule = pd.DataFrame()
    for index in schedule_with_change_over.index[1::2]: # pass through every jumbo
        order_number = schedule_with_change_over.iloc[index]['Order Number']

        ### these are columns specific to width layouts
        match = '\d\d\d'
        layout_columns = [i for i in schedule_with_change_over.columns if re.match(match, i)]


        deckle_layout =  deckle_order = pd.DataFrame(schedule_with_change_over.iloc[index]).T

        ### these are columns specific to order details
        order_columns = [i for i in deckle_order.columns if (i not in layout_columns)
                         & (i not in ['freq', 'layout number'])]

        ### for order we grab the last row since this is the real 'completion date'
        order = deckle_order[order_columns].reset_index(drop=True).iloc[-1].dropna()

        ### remove widths that are in the deckle opt. but not in the layout
        formula = deckle_layout[layout_columns].reset_index(drop=True).iloc[0].dropna()

        ### number of doffs for the given layout/order
        doffs = doffs_in_jumbo
        read_out = ''
        for i in range(formula.shape[0]-1): #for unique prods
            read_out = read_out + ("{}x{} + ".format(formula.index[i],
                                                    formula.iloc[i].astype(int)))
        read_out = read_out + ("{}x{}".format(formula.index[-1],
                                            formula.iloc[-1].astype(int))) #make last string w/o +
        current = pd.DataFrame([read_out, int(doffs), order]).T
        order = pd.DataFrame(order).T
        order = order.reset_index(drop=True)
        current = pd.DataFrame([read_out, int(doffs)]).T
        current.columns = ['Formula', 'Doffs']
        current = current.join(order)

        ### add changeover info
        current_changeover = pd.DataFrame(current.columns)
        current_changeover = current_changeover.set_index(0)
        current_changeover = current_changeover.T
        current_changeover = current_changeover.append({'Order Number': 'Changeover'}, ignore_index=True)
        current_changeover['Completion Date'] = schedule_with_change_over.iloc[index-1]['Completion Date']
        current_combined = current_changeover.append(current, sort=False, ignore_index=True)
        current_combined

        master_schedule = pd.concat([master_schedule, current_combined], sort=False)
        master_schedule = master_schedule.reset_index(drop=True)
    return master_schedule
