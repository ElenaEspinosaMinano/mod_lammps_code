#!/bin/bash
# bash script for trimming outfiles and cs outfiles for original outfiles of models 5, 6, 7 + 8 with increasing no of proteins

# loop for range 5 to 8 - Models 5, 6, 7 + 8
# then loop for range 3 to 6 for increasing no of proteins
for i in {5..8}
do
    for j in {3..6}
    do
        nohup python3 trim_outfile.py outfile_${i}_var_${j}00_run_3_v2.dat trimmed_outfile_${i}_var_${j}00_run_3_v2.dat 3000000 & disown
        nohup python3 trim_cs_outfile.py outfile_cs_${i}_var_${j}00_run_3_v2.dat trimmed_outfile_cs_${i}_var_${j}00_run_3_v2.dat 3000000 & disown
    done
done

# wait for all background processes to finish
wait
