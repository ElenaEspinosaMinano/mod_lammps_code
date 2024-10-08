#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for dump files of models 1, 2, 3, 4

# loop for range 1 to 3 - Models 1, 2 + 3
for i in {1..3}
do
    nohup python3 main_calculations.py dump_model_${i}_run_1.dat outfile_${i}_run_1.dat outfile_cs_${i}_run_1.dat & disown
done

# loop for range 1 to 8 - Model 4 (and control) for attraction strength: 1-8 kBT (run 1)
for j in {1..8}
do
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_1.dat outfile_4_var_${j}_run_1.dat outfile_cs_4_var_${j}_run_1.dat & disown
    nohup python3 main_calculations.py dump_model_4_var_${j}_run_1_control.dat outfile_4_var_${j}_run_1_control.dat outfile_cs_4_var_${j}_run_1_control.dat & disown
done

# loop for range 1 to 13 - Model 4 (and control) for attraction strength: 0.3-3.9 kBT (run 2)
for k in {9..13}
do
    nohup python3 main_calculations.py dump_model_4_var_${k}_run_2.dat outfile_4_var_${k}_run_2.dat outfile_cs_4_var_${k}_run_2.dat & disown
    nohup python3 main_calculations.py dump_model_4_var_${k}_run_2_control.dat outfile_4_var_${k}_run_2_control.dat outfile_cs_4_var_${k}_run_2_control.dat & disown
done

# wait for all background processes to finish
wait
