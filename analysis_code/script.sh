#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for all dump files

# loop for range 1 to 3 - Models 1, 2 + 3
for i in {1..3}
do
    nohup python3 main_calculations.py dump_model_${i}_run_1.dat outfile_${i}_run_1.dat outfile_cs_${i}_run_1.dat &
done

# loop for range 1 to 8 - Model 4 (and control) for attraction strength: 1-8 kBT (run 1 ) and 0.3-3.9 kBT (run 2)
for j in {1..8}
do
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_1.dat outfile_4_var_${j}_run_1.dat outfile_cs_4_var_${j}_run_1.dat &
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_1_control.dat outfile_4_var_${j}_run_1_control.dat outfile_cs_4_var_${j}_run_1_control.dat &
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_2.dat outfile_4_var_${j}_run_2.dat outfile_cs_4_var_${j}_run_2.dat &
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_2_control.dat outfile_4_var_${j}_run_2_control.dat outfile_cs_4_var_${j}_run_2_control.dat &
done

# wait for all background processes to finish
wait
