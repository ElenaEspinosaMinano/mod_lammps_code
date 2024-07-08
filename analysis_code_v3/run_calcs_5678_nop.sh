#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for dump files of models 5, 6, 7 + 8 with increasing no of proteins

# loop for range 5 to 8 - Models 5, 6, 7 + 8
# then loop for range 3 to 5 for increasing no of proteins
for i in {5..8}
do
    for j in {3..5}
    do
        nohup python3 main_calcs_05678.py dump_model_${i}_var_${j}00_run_1.dat outfile_${i}_var_${j}00_run_1_v3.dat outfile_cs_${i}_var_${j}00_run_1_v3.dat 5${j}00 & disown
    done
done

# wait for all background processes to finish
wait
