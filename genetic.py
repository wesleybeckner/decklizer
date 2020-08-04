import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from random import shuffle, choice
import copy
from utils import *
from IPython.display import clear_output
import time

# df = pd.read_excel('../../data/berry/200721_ New Way SCH W3-W6 W14 07.20.20.xlsx',
#                    sheet_name='WAY-14  new cycle system (3)')
# df = df.loc[df['Customer Name'] == 'P & G']
# df = df.loc[df['Description'].str.contains('SAM')]
# df = df.loc[df['Description'].str.contains('WHITE')] #CYAN, TEAL
# df = df.loc[df['CYCLE / BUCKET'] == 'CYCLE 1']
# df = df.reset_index(drop=True)
# df['Width'] = pd.DataFrame(list(pd.DataFrame(list(df['Description'].str.split(';')))[0].str.split('UN0')))[1]
# lm = list(df.groupby('Description')['Total LM Order QTY'].sum().values)
# widths = list(df.groupby('Description')['Width'].first().values.astype(int))
# B = 4160
# L = 17000 # df['LM putup']
# neckin = [4, 5, 7, 7, 7] # 158 MM missing cycle 1 4 mm knife in
# w = list(np.array(widths) + np.array(neckin))
# q = [math.ceil(x/L) for x in lm]

# s = BinPackingExample(w, q)

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
        print(dff)
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
        clear_output(wait=True)
        print("gene time: {:.1f}, chromosome time: {:.1f}, ffd time: {:.1f}"
              .format(gene_time, chrome_time, ffd_time))
        print("loss: {:.3f}, best loss: {:.3f}, steps: {}, gene count: {}, unique doffs: {}, max unique on doff: {}"
              .format(loss, best_loss, step, gene_count,
                      summarize_results(sol_tot, widths, neckin, B).shape[0],
                      max(pd.DataFrame(sol_tot).replace(0, np.nan, inplace=False)\
                      .nunique(axis=1))))

        if (loss < loss_target) and \
            (summarize_results(sol_tot, widths, neckin, B).shape[0] < 20) and \
            (all(pd.DataFrame(sol_tot).replace(0, np.nan, inplace=False)\
            .nunique(axis=1) <= max_unique_products)):
            break
    print(summarize_results(sol_tot, widths, neckin, B))
    return sol_tot, loss
