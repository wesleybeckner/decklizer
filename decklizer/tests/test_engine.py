import sys
sys.path.append("../")
import decklizer as dl


def test_engine():
    specs = [['P & G', 'SAM'], ['AHP', 'SM']]
    max_patterns_options = [3, 4]
    max_combinations=3
    for spec in specs:
        for max_patterns in max_patterns_options:
            df = dl.load_schedule(
            customer=spec[0],
            technology=spec[1],
            color='WHITE',
            cycle='CYCLE 2')

            B = 4160 # usable width
            w, q, L, n = dl.process_schedule(
                df,
                B=B,
                put_up=17000,
                doffs_in_jumbo=6,
                verbiose=True,
            )

            patterns, layout = dl.seed_patterns(w, q, B, n,
                            max_combinations=max_combinations, verbiose=True)

            loss, inventory, summary = dl.find_optimum(patterns,
                            layout, w, q, B, n, L,
                            max_combinations=max_combinations,
                            max_patterns=max_patterns,
                            prioritize='time')
