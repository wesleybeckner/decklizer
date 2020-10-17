import numpy as np
import pandas as pd
import math
from random import shuffle, choice
import copy
# from utils import *
import time
from collections import Counter
import itertools
from scipy.optimize import linprog

def seed_patterns(w, q, B, n, max_combinations=3, verbiose=True):
    layout = make_best_pattern(q, w, n, verbiose=verbiose)
    combos = list(itertools.combinations(w,r=max_combinations))
    if verbiose:
        print('')
        print("{} possible max {} combinations".format(len(combos),max_combinations))
    patterns = []
    for combo in combos:

        # only provide knapsack with relevant variables
        s = []
        for i in combo:
            s += (int(B/i)*[i])
        t = initt(B,len(s))
        knapsack(s, s, B, len(s), t)
        t = np.array(t)
        patterns += store_patterns(t, s, B, goal=3)
    uni_list = []
    for i in patterns:
        if i not in uni_list:
            uni_list.append(i)
    patterns = uni_list
    patterns = list(np.array(patterns)[np.array(patterns)[:,1]>=0])
    if verbiose:
        print("{} unique patterns found".format(len(patterns)))
    return patterns, layout
    
def knapsack(wt, val, W, n, t):

    # base conditions
    if n == 0 or W == 0:
        return 0
    if t[n][W] != -1:
        return t[n][W]

    # choice diagram code
    if wt[n-1] <= W:
        t[n][W] = max(
            val[n-1] + knapsack(
            wt, val, W-wt[n-1], n-1, t),
            knapsack(wt, val, W, n-1, t))
        return t[n][W]
    elif wt[n-1] > W:
        t[n][W] = knapsack(wt, val, W, n-1, t)
        return t[n][W]

def store_patterns(t, s, B, goal=5):
    patterns = []
    bit = 0
    while len(patterns) < goal:
        found = 0
        for pair in np.argwhere(t == t.max()-bit):
            N, W = pair
            sack = reconstruct(N, W, t, s)
            pattern = Counter(np.array(s)[list(sack)])
            loss = round((B - np.array(s)[list(sack)].sum())/B*100,2)
            patterns.append([pattern, loss])
            if len(patterns) >= goal:
                break
            found += 1
            if found > 1:
                break
        bit += 2
    return patterns

def reconstruct(i, w, kp_soln, weight_of_item):
    """
    Reconstruct subset of items i with weights w. The two inputs
    i and w are taken at the point of optimality in the knapsack soln

    In this case I just assume that i is some number from a range
    0,1,2,...n
    """
    recon = set()
    # assuming our kp soln converged, we stopped at the ith item, so
    # start here and work our way backwards through all the items in
    # the list of kp solns. If an item was deemed optimal by kp, then
    # put it in our bag, otherwise skip it.
    for j in range(i)[::-1]:
        cur_val = kp_soln[j][w]
        prev_val = kp_soln[j-1][w]
        if cur_val > prev_val:
            recon.add(j)
            w = w - weight_of_item[j]
    return recon

def initt(W, n):
    # We initialize the matrix with -1 at first.
    return [[-1 for i in range(W + 1)] for j in range(n + 1)]

def make_best_pattern(q, w, n, usable_width=4160, verbiose=True):
    """
    Creates the best possible pattern such that all orders are fullfilled in a single
    layout

    Parameters
    ----------
    q: list
        rolls required (in jumbo lengths)
    w: list
        widths required
    n: list
        neckins for widths
    usable_width: int
        jumbo/doff usable width

    Returns
    -------
    layout: list
        cuts for jumbo for each width (no width is excluded)
    """
    layout = [max(1, math.floor(i/sum(q)*usable_width/j)) for i,j in zip(q,w)]

    # give priority to widths that had to round down the most
    # when filling up the rest of the pattern
    remainder = [math.remainder(i/sum(q)*usable_width/j, 1) if (math.remainder(i/sum(q)*usable_width/j, 1)
                                                        < 0) else -1 for i,j in zip(q,w) ]
    order = np.argsort(remainder)
    while (usable_width - sum([i*j for i,j in zip(layout,w)])) > min(w):
        for i in order[::-1]:
            layout[i] += 1
            if usable_width - sum([i*j for i,j in zip(layout,w)]) < 0:
                layout[i] -= 1

    # compute the loss for the final layout
    layout_loss = usable_width - sum([i*j for i,j in zip(layout,w)])
    if verbiose:
        print("layout pattern: {}".format(dict(zip([i-j for i,j in zip(w,n)],layout))))
        print("pattern loss: {:0.2f} %".format(layout_loss/usable_width*100))

    # multiply to get the minimum doffs required
    # layout * doffs > q
    doffs = max([math.ceil(i/j) for i,j in zip(q, layout)])
    if verbiose:
        print("minimum doffs to fill order: {}".format(doffs))

    # what inventory is created
    inventory = dict(zip([i-j for i,j in zip(w,n)],[i*doffs-j for i,j in zip(layout,q)]))
    if verbiose:
        print("inventory created: {}".format(inventory))

    return layout

def init_layouts(B, w):
    t = []
    m = len(w)
    for i in range(m):
        pat = [0]*m
        pat[i] = -int(B/w[i])
        t.append(pat)
    return t



def output_results(result, lhs_ineq, B, w, n, q, L):
    sheet = np.sum([(i*j) for i,j in zip(w, np.array(lhs_ineq))],axis=0)#*np.ceil(result['x'])
    inventory = dict(zip([i-j for i,j in zip(w,n)],np.sum(np.array(lhs_ineq)*-1*np.ceil(result['x']),axis=1)-np.array(q)))

    # create layout summary
    jumbos = list(np.ceil(result['x'])[np.ceil(result['x'])>0])
    temp = np.array(lhs_ineq)*-1*np.where(np.ceil(result['x']) != 0, 1, 0)
    temp = temp[:, temp.any(0)].T
    non_zero_layouts = list([dict(zip([i-j for i,j in zip(w,n)], i)) for i in temp])

    sheet_loss = [B+i for i in sheet]
    sheet_loss = [i / B * 100 for i,j in zip(sheet_loss,np.where(result['x'] > 0, 1, 0)) if j > 0]

    # remove extra layouts due to ceiling rounding from linprog
    summary = pd.DataFrame([sheet_loss, jumbos, non_zero_layouts]).T
    summary.columns = ['loss', 'jumbos', 'layout']
    summary = summary.sort_values('loss', ascending=False).reset_index(drop=True)
    for index, layout2 in enumerate(summary['layout']):
        if all(np.array(list(inventory.values())) - np.array(list(layout2.values())) > 0):
            summary.loc[index, 'jumbos'] -= 1
            new_values = np.array(list(inventory.values())) - np.array(list(layout2.values()))
            inventory.update(zip(inventory,new_values))
    summary = summary[summary['jumbos'] != 0]

    loss = sum([i[0]*i[1] for i in summary.values])/sum([i[1] for i in summary.values])
    sqm_inventory = np.sum([i*j*.001*L for i,j in zip (inventory.keys(),inventory.values())])
    sqm_produced = np.sum(jumbos)*L*B*.001
    sqm_loss = sqm_produced*loss/100

    print("total loss:      {:0.2f} % ({:.2e} sqm)".format(loss, sqm_loss))
    print("total inventory: {:.2f} % ({:.2e} sqm)".format(sqm_inventory/sqm_produced*100, sqm_inventory), end = '\n\n')
    print("inventory created: {}".format(inventory), end = '\n\n')
    # print("total inventory rolls: {:n} ({:.2e} sqm)".format(sum(list(inventory.values())), sqm_inventory), end='\n\n')
    print("layout summary:", end = '\n\n')
    for i in summary.values:
        print("loss: {:.2f}% \t {} x\t {}".format(i[0], i[1], i[2]))
    print('')
    print("total jumbos: {} ({:.2e} sqm)".format(np.sum(summary['jumbos']), sqm_produced))

    return loss, inventory, summary

# choose max unique widths per doff
def find_optimum(patterns, layout, w, q, B, n, L, max_combinations=3, max_patterns = 3, prioritize = 'time'):
    if prioritize == 'material loss':
        inv_loss = 0
    elif prioritize == 'time':
        inv_loss = 1
    if 1 < max_patterns < 4:
        # find best of X combination
        if len(w) <= max_combinations:
            pattern_combos = list(itertools.combinations(patterns,r=max_patterns-1))
        else:
            pattern_combos = list(itertools.combinations(patterns,r=max_patterns))
        print("{} possible max {} patterns".format(len(pattern_combos),max_patterns), end='\n\n')
        best_of = []
        for combo in pattern_combos:
            patterns2 = combo
            lhs_ineq = []
            for pattern in patterns2:
                inset = []
                for width in w:
                    try:
                        inset.append(-pattern[0][width])
                    except:
                        inset.append(0)
                lhs_ineq.append(inset)
        #     naive = init_layouts(B, w)
        #     lhs_ineq = lhs_ineq + naive
            if len(w) <= max_combinations:
                lhs_ineq.append([-i for i in layout])
            lhs_ineq = np.array(lhs_ineq).T.tolist()
            rhs_ineq = [-i for i in q]
            obj = np.ones(len(lhs_ineq[0]))

            result = linprog(c=obj,
                    A_ub=lhs_ineq,
                    b_ub=rhs_ineq,
                    method="revised simplex")
            if result['success'] == True:
                sheet = np.sum([(i*j) for i,j in zip(w, np.array(lhs_ineq))],axis=0)#*np.ceil(result['x'])
                inventory = dict(zip([i-j for i,j in zip(w,n)],np.sum(np.array(lhs_ineq)*-1*\
                                                                      np.ceil(result['x']),axis=1)-np.array(q)))

                # create layout summary
                jumbos = list(np.ceil(result['x'])[np.ceil(result['x'])>0])
                temp = np.array(lhs_ineq)*-1*np.where(np.ceil(result['x']) != 0, 1, 0)
                temp = temp[:, temp.any(0)].T
                non_zero_layouts = list([dict(zip([i-j for i,j in zip(w,n)], i)) for i in temp])

                sheet_loss = [B+i for i in sheet]
                sheet_loss = [i / B * 100 for i,j in zip(sheet_loss,np.where(result['x'] > 0, 1, 0)) if j > 0]

                # remove extra layouts due to ceiling rounding from linprog
                sorted_jumbos = [x for _,x in sorted(zip(sheet_loss,jumbos))][::-1]
                sorted_layouts = np.array(non_zero_layouts)[np.array(sheet_loss).argsort()][::-1]
                sorted_losses = [x for _,x in sorted(zip(sheet_loss,sheet_loss))][::-1]
                for index, layout2 in enumerate(np.array(non_zero_layouts)[np.array(sheet_loss).argsort()][::-1]):
                    if all(np.array(list(inventory.values())) - np.array(list(layout2.values())) > 0):
                        sorted_jumbos[index] -= 1
                        new_values = np.array(list(inventory.values())) - np.array(list(layout2.values()))
                        inventory.update(zip(inventory,new_values))

                        # clear layouts that have been set to 0
                summary = (list(zip(sorted_jumbos, sorted_layouts, sorted_losses)))
                summ = []
                for i in summary:
                    if i[0] > 0:
                        summ.append(i)
                summary=summ
                loss = sum([i[0]*i[2] for i in summary])/sum([i[0] for i in summary])

                best_of.append([loss, sum(list(inventory.values())), patterns2])

        # minimize inventory or minimize mat. loss
        arr = np.array(best_of, dtype=object)
        patterns_final = arr[np.argmin(arr[:,inv_loss])][2]
    elif max_patterns == 1:
        patterns_final = [[dict(zip(w,layout)), 0]]

    else:
        patterns_final = patterns

    # find overall best combination
    # format layouts for linear optimization
    lhs_ineq = []
    for pattern in patterns_final:
        inset = []
        for width in w:
            try:
                inset.append(-pattern[0][width])
            except:
                inset.append(0)
        lhs_ineq.append(inset)
    # naive = init_layouts(B, w)
    # lhs_ineq = lhs_ineq + naive
    if len(w) <= max_combinations:
        lhs_ineq.append([-i for i in layout])
    lhs_ineq = np.array(lhs_ineq).T.tolist()
    rhs_ineq = [-i for i in q]
    obj = np.ones(len(lhs_ineq[0]))

    result = linprog(c=obj,
            A_ub=lhs_ineq,
            b_ub=rhs_ineq,
            method="revised simplex")

    return output_results(result, lhs_ineq, B, w, n, q, L)

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

def find_optimum_mc(s,
                 B,
                 widths,
                 neckin,
                 loss_target = 2,
                 max_doff_layouts = 20,
                 max_unique_products = 6,
                 gene_count = 5):
    """
    Parameters
    ----------
    s: list
        list of widths for each order
    B: int
        usable doff width
    widths: list
        list of unique widths
    neckin: list
        list of neckin for each width
    loss_target: float or int
        minimization goal for material loss
    max_doff_layouts: int
        maximum allowable unique layouts
    max_unique_products: int
        maximum allowable unique products per layout
    gene_count: int
        hyperparameter evolutionary algorithm. Number of layouts to pull from
        initial knapsack packing

    Returns
    -------
    sol: list
        list of optimized layouts
    loss: float
        total fractional material loss for all layouts
    """
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
                # for mult in range(1):

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
                    # for mult in range(doff_min):
                    sol2.append(new_gene)

        chrome_time = time.time() - gene_time
        #     if step > iterations:
        #         break
        remain2 = [B] #initialize list of remaining bin spaces
        sol3 = [[]]
        binlim = np.inf
        # doff_min = 2
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
    sol = sol_tot
    return sol, loss

def optimize_schedule(sol, widths, neckin, df_filtered, L, setup_df, speed_df,
                      doffs_in_jumbo, start_date_time, objective='Time (Knife Changes)', DEBUG=False):

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
    elif objective == "Time (Knife Changes)":
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
                        if extras[width][0] >= doffs:

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
    else:
        print("no valid objective provided")
        return 0

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
    return master_schedule, extras
