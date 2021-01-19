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

def seed_patterns(w, q, B, n, max_combinations=3, goal=3, verbiose=True):
    '''
    creates a number of optimal patterns for deckling

    Parameters
    ----------
    w: list
        list of widths (int)
    q: list
        list of rolls for each width (int)
    B: int
        usuable width per mother roll
    n: list
        neck in for each width (int)
    max_combinations: int, default 3
        maximum number of unique products (widths) to have on a mother roll
    goal: int, default 3
        the desired number of recovered patterns from the knapsack problem
        for every unique grouping of unique widths at max_combinations
    verbiose: bool, default True
        turns on/off print statements during execution

    Returns
    -------
    patterns: list of lists
        list of pattern, loss pairs. Pattern is a dictionary containing a width,
        count pair that describes the pattern on the mother roll. Loss is the
        percent material loss for the pattern.
    layout: list
        list of counts for every width on the mother roll. Layout is the best
        possible pattern in terms of minimizing mother rolls to create the order
        with a single pattern.
    '''
    layout = make_best_pattern(q, w, n, B, verbiose=verbiose)
    combos = []
    for i in range(1,max_combinations+1)[::-1]:
        combos += list(itertools.combinations(w,r=i))
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
        patterns += store_patterns(t, s, B, goal=goal)
        for j in range(3):
            for i in patterns:
                for key in list(i[0].keys()):
                    loss = (B - np.sum(np.array(list(i[0].keys())) *
                        np.array(list(i[0].values())))) - key
                    if loss > 0:
                        i[0][key] += 1
                        i[1] = loss
    uni_list = []
    for i in patterns:
        if i not in uni_list:
            uni_list.append(i)
    patterns = uni_list
    patterns = list(np.array(patterns)[np.array(patterns)[:,1]>=0])

    # the naive patterns should be kept due to their usefullness
    # in order fulfilment regardless of loss
    naive = init_layouts(B, w)
    for i in naive:
        i = [-j for j in i]
        patterns.append([dict(zip(w,i)),0])

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
            loss = B - np.sum(np.array(list(pattern.keys())) *
                            np.array(list(pattern.values())))
            if loss > 0:
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
    # sometimes the floor still puts us over
    for i in range(3):
        if usable_width - sum([i*j for i,j in zip(layout,w)]) < 0:
            layout[np.argmax(layout)] -= 1

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

    # sometimes all orders can't be fullfilled in a single layout
    if any([i == 0 for i in layout]):
        return layout
    else:
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
def find_optimum(patterns, layout, w, q, B, n, L, max_combinations=3,
                 max_patterns = 3, prioritize = 'time', qt=None):
    '''
    Finds the best possible slitter schedule by linear optimization of a set of
    patterns.

    Parameters
    ----------
    patterns: list of lists
        list of pattern, loss pairs. Pattern is a dictionary containing a width,
        count pair that describes the pattern on the mother roll. Loss is the
        percent material loss for the pattern.
    layout: list
        list of counts for every width on the mother roll. Layout is the best
        possible pattern in terms of minimizing mother rolls to create the order
        with a single pattern.
    w: list
        list of widths (int)
    q: list
        list of rolls for each width (int)
    B: int
        usuable width (mm) of mother roll
    n: list
        neck in for each width (int)
    L: int
        length (m) of mother roll
    max_combinations: int, default 3
        maximum number of unique products (widths) to have on a mother roll
    max_patterns: int, default 3
        maximum number of patterns for deckle schedule
    prioritize: str, default 'time'
        only relevant when max_patterns < 4. When max_patterns < 4 either the
        lowest material waste linear opimization of patterns is returned
        ('material loss') or the fewest mother rolls used is returned ('time')
    qt: list
        default: None. List of the true values for q if algorithm is called
        using an adjusted production target. This is used to set the output
        results inventory to the true inventory created as a result of the
        true order

    Returns
    -------
    loss: float
        percent loss for the deckle schedule
    inventory: dictionary
        width, count pairs that describe the inventory created above the order
        critieria
    summary: dataframe
        dataframe of loss, jumbos, layout that describe the percent material
        loss, total number of jumbos (or doffs), and the dictionary of width,
        count pairs that describe the pattern for every pattern in the deckle
        schedule
    '''
    if qt == None:
        qt = q
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
    if result['success'] == False:
        print('Error')
        print(result['message'])
        return 0
    return output_results(result, lhs_ineq, B, w, n, qt, L)

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
