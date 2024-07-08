#!/bin/bash
# bash script for trimming outfiles and cs outfiles of dump files models 5678

# loop for range 5 to 6 - Models 5 + 6
for i in {5..7}
do
    nohup python3 trim_outfile.py outfile_${i}_run_1_v3.dat trimmed_outfile_${i}_run_1_v3.dat 3000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_${i}_run_1_v3.dat trimmed_outfile_cs_${i}_run_1_v3.dat 3000000 & disown
done

# loop for range 3 to 7 - Model 7 with varying switching rate: 300-700
for j in {3..7}
do
    nohup python3 trim_outfile.py outfile_7_sw_${j}00_run_1_v3.dat trimmed_outfile_7_sw_${j}00_run_1_v3.dat 3000000 & disown
    nohup python3 trim_cs_outfile.py outfile_cs_7_sw_${j}00_run_1_v3.dat trimmed_outfile_cs_7_sw_${j}00_run_1_v3.dat 3000000 & disown
done

# Model 0 (NO attraction between any polymer types with protein switching rate 500)
nohup python3 trim_outfile.py outfile_0_run_1_v3.dat trimmed_outfile_0_run_1_v3.dat 3000000 & disown
nohup python3 trim_cs_outfile.py outfile_cs_0_run_1_v3.dat trimmed_outfile_cs_0_run_1_v3.dat 3000000 & disown

# Model 8 (model 7 with switching rate 500 and protein-protein attraction strength 2 kBT)
nohup python3 trim_outfile.py outfile_8_run_1_v3.dat trimmed_outfile_8_run_1_v3.dat 3000000 & disown
nohup python3 trim_cs_outfile.py outfile_cs_8_run_1_v3.dat trimmed_outfile_cs_8_run_1_v3.dat 3000000 & disown

# wait for all background processes to finish
wait
