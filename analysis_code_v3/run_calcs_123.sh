#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for dump files of models 1, 2, 3, 4

# loop for range 1 to 3 - Models 1, 2 + 3
for i in {1..3}
do
    nohup python3 main_calcs_1234.py dump_model_${i}_run_1.dat outfile_${i}_run_1_v3.dat outfile_cs_${i}_run_1_v3.dat & disown
done

# wait for all background processes to finish
wait
