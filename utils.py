import numpy as np
import pandas as pd
import math
from random import shuffle
import urllib
import copy

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

def optimize_late_orders(sol, widths, neckin, df, L):
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
    extras.columns = master2.columns[:-1]
    schedule = []
    layout_pattern = 0
    old_width = width = None
    for row1 in df.index:
        current = df.iloc[row1][['Total LM Order QTY', 'Width', 'Scheduled Ship Date']]
        doffs = math.ceil(current['Total LM Order QTY'] / L) # QTY
        width = current['Width']
        if width == old_width:
            layout_pattern -= 1
        master2 = master2.sort_values(width, ascending=False).reset_index(drop=True)
        target_doffs = extras.iloc[0][width]
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
        print(extras)
        clear_output(wait=True)

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
