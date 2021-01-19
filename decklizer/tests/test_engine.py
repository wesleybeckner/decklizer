import sys
sys.path.append("../")
import decklizer as dl
import multiprocessing as mp
import numpy as np
import pandas as pd
import math

def test_engine():
    buckets = dict()
    buckets['A'] = dict()
    buckets['A']['w'] = [i+j for i, j in zip([205, 195, 220, 160],
                                             [3, 2, 3, 1])]
    buckets['A']['n'] = [3, 2, 3, 1]
    buckets['A']['L'] = 17000
    buckets['A']['q'] = [776, 557, 470, 17]
    buckets['A']['B'] = 4160

    buckets['B'] = dict()
    buckets['B']['w'] = [i+j for i, j in zip([575, 626, 438, 622, 749, 546],
                                             [10, 11, 9, 11, 14, 10])]
    buckets['B']['n'] = [10, 11, 9, 11, 14, 10]
    buckets['B']['L'] = 15000
    buckets['B']['q'] = [282, 282, 5, 48, 142, 241]
    buckets['B']['B'] = 3050

    buckets['C'] = dict()
    buckets['C']['w'] = [i+j for i, j in zip([622, 749, 800],
                                             [11, 14, 14])]
    buckets['C']['n'] = [11, 14, 14]
    buckets['C']['L'] = 15000
    buckets['C']['q'] = [175, 495, 330]
    buckets['C']['B'] = 3050

    buckets['D'] = dict()
    buckets['D']['w'] = [i+j for i, j in zip([616, 743],
                                             [11, 14])]
    buckets['D']['n'] = [11, 14]
    buckets['D']['L'] = 15000
    buckets['D']['q'] = [850, 150]
    buckets['D']['B'] = 3050

    params = []
    combination_params = [2,3]
    pattern_params = [3,4]
    production_targets = np.arange(0.97,1.03,.01)
    edge_trim_allowance = np.arange(1)
    goal = 3
    verbiose = False
    bucket = buckets['A']
    res = []
    for max_combo in combination_params:
        for max_patterns in pattern_params:
            for target in production_targets:
                for edge in edge_trim_allowance:
                    params.append([bucket, max_combo, max_patterns, target,
                        edge, goal, verbiose])
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(deckle, args=(param[0],
                                        param[1],
                                        param[2],
                                        param[3],
                                        param[4],
                                        param[5],
                                        param[6])) for param in params]
    pool.close()
    while True:
        if all([i.ready() for i in results]):
            res = []
            for i in results:
                res.append(i.get())

            df = pd.DataFrame(res, columns=['loss', 'jumbos', 'inventory', 'summary',
                                    'combinations', 'patterns', 'target', 'edge'])

            df['str summ'] = df['summary'].astype(str)
            df = df.sort_values('loss').reset_index(drop=True)
            df = df.drop_duplicates(['str summ'])\
                [['loss', 'jumbos', 'inventory', 'summary',
                 'combinations', 'patterns', 'target', 'edge']].reset_index(drop=True)
            df['loss rank'] = df['loss'].rank(pct=True)
            df['jumbo rank'] = df['jumbos'].rank(pct=True)

            break
    print(df)


def deckle(bucket, max_combo, max_patterns, target, edge, goal, verbiose):
    patterns, layout = dl.seed_patterns(
        bucket['w'],
        [math.ceil(i * target)
        for i in bucket['q']],
        bucket['B'] + (edge * 2),
        bucket['n'],
        max_combinations=max_combo,
        goal=goal,
        verbiose=verbiose)
    loss, inv, summary = dl.find_optimum(
        patterns,
        layout,
        bucket['w'],
        [math.ceil(i * target)
        for i in bucket['q']],
        bucket['B'] + (edge * 2),
        bucket['n'],
        bucket['L'],
        max_combinations=max_combo,
        max_patterns=max_patterns,
        prioritize='time',
        qt=bucket['q'])
    return [loss, int(summary['jumbos'].sum()),
            inv, summary, max_combo,
            summary.shape[0], target, edge]
