#!/bin/bash
# bash script for trimming outfiles and cs outfiles of all dump files

# loop for range 1 to 3 - Models 1, 2 + 3
for i in {1..3}
do
    nohup python3 trim_outfile.py outfile_${i}_run_1.dat trimmed_outfile_${i}_run_1.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_${i}_run_1.dat trimmed_outfile_cs_${i}_run_1.dat 1000000 & disown
done

# loop for range 1 to 8 - Model 4 (and control) for attraction strength: 1-8 kBT (run 1)
for j in {1..8}
do
    nohup python3 trim_outfile.py outfile_4_var_${j}_run_1.dat trimmed_outfile_4_var_${j}_run_1.dat 1000000 & disown
    nohup python3 trim_outfile.py outfile_4_var_${j}_run_1_control.dat trimmed_outfile_4_var_${j}_run_1_control.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_4_var_${j}_run_1.dat trimmed_outfile_cs_4_var_${j}_run_1.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_4_var_${j}_run_1_control.dat trimmed_outfile_cs_4_var_${j}_run_1_control.dat 1000000 & disown
done

# loop for range 1 to 13 - Model 4 (and control) for attraction strength: 0.3-3.9 kBT (run 2)
for k in {9..13}
do
    nohup python3 trim_outfile.py outfile_4_var_${k}_run_2.dat trimmed_outfile_4_var_${k}_run_2.dat 1000000 & disown
    nohup python3 trim_outfile.py outfile_4_var_${k}_run_2_control.dat trimmed_outfile_4_var_${k}_run_2_control.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_4_var_${k}_run_2.dat trimmed_outfile_cs_4_var_${k}_run_2.dat 1000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_4_var_${k}_run_2_control.dat trimmed_outfile_cs_4_var_${k}_run_2_control.dat 1000000 & disown
done

# wait for all background processes to finish
wait
