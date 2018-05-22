#!/bin/bash
# Ren Zhang @ ryanzjlib dot gmail dot com
cd python
python rolling_summaries.py --mode=pred --pred_bj_windows=golden_5 --pred_ld_windows=fib_9 --save=True --submit=True
python prophet.py --mode=pred --save=True --submit=True
python shortcut_mlp.py --mode=pred --save=True --submit=True