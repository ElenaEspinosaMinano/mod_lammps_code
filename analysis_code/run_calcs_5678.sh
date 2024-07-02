#!/bin/bash
# bash script for obtaining outfiles and cs outfiles for dump files of models 5, 6, 7, 8

# loop for range 5 to 6 - Models 5 + 6
for i in {5..6}
do
    nohup python3 main_calculations.py dump_model_${i}_run_1.dat outfile_${i}_run_3_v2.dat outfile_cs_${i}_run_3_v2.dat & disown
done

# loop for range 3 to 7 - Model 7 with varying switching rate: 300-700
for j in {3..7}
do
    nohup python3 main_calculations.py dump_model_7_sw_${j}00_run_1.dat outfile_7_sw_${j}00_run_3_v2.dat outfile_cs_7_sw_${j}00_run_3_v2.dat & disown
done

# Model 8 (model 7 with switching rate 500 and protein-protein attraction strength 2 kBT)
nohup python3 main_calculations.py dump_model_8_run_1.dat outfile_8_run_3_v2.dat outfile_cs_8_run_3_v2.dat & disown


# wait for all background processes to finish
wait
