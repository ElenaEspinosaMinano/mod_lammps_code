#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for dump files of models 5, 6, 7, 8

# loop for range 5 to 6 - Models 5 + 6
for i in {5..6}
do
    nohup python3 main_calcs_05678.py dump_model_${i}_run_1.dat outfile_${i}_run_1_v3.dat outfile_cs_${i}_run_1_v3.dat 5600 & disown
done

# loop for range 3 to 7 - Model 7 with varying switching rate: 300-700
for j in {3..7}
do
    nohup python3 main_calcs_05678.py dump_model_7_sw_${j}00_run_1.dat outfile_7_sw_${j}00_run_1_v3.dat outfile_cs_7_sw_${j}00_run_1_v3.dat 5600 & disown
done

# Model 0 (NO attraction between any polymer types with protein switching rate 500)
nohup python3 main_calcs_05678.py dump_model_0_run_1.dat outfile_0_run_1_v3.dat outfile_cs_0_run_1_v3.dat 5600 & disown

# Model 8 (model 7 with switching rate 500 and protein-protein attraction strength 2 kBT)
nohup python3 main_calcs_05678.py dump_model_8_run_1.dat outfile_8_run_1_v3.dat outfile_cs_8_run_1_v3.dat 5600 & disown

# wait for all background processes to finish
wait
