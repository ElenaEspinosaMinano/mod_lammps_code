#!/bin/bash
# bash script for trimming outfiles and cs outfiles of dump files models 123

# loop for range 1 to 3 - Models 1, 2 + 3
for i in {1..3}
do
    nohup python3 trim_outfile.py outfile_${i}_run_1_v3.dat trimmed_outfile_${i}_run_1_v3.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_${i}_run_1_v3.dat trimmed_outfile_cs_${i}_run_1_v3.dat 1000000 & disown
done

# wait for all background processes to finish
wait
