import sys
sys.path.append("../")
import decklizer as dl


def test_engine():
    specs = [['P & G', 'SAM'], ['AHP', 'SM']]
    for spec in specs:
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
        max_combinations=3
        patterns, layout = dl.seed_patterns(w, q, B, n,
                        max_combinations=max_combinations, verbiose=True)

        loss, inventory, summary = dl.find_optimum(patterns,
                        layout, w, q, B, n, L,
                        max_combinations=max_combinations, max_patterns=4,
                        prioritize='time')
