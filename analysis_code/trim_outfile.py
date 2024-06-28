### Trim transient part of outfile

import os
import sys

name_outfile = sys.argv[1] # name of outfile to trim - should be a COPY in trimmed_outfiles directory
name_trimmed_outfile = sys.argv[2] # name of final trimmed outfile
start_timestep = int(sys.argv[3]) # steady state starts from timestep - 1x10^6 (in model 1234 sims) 3x10^6 (in model 567 sims)

# specify path
path_to_trimmed_outfiles = '../outfiles/trimmed_outfiles/'

with open(path_to_trimmed_outfiles + name_outfile, 'r+') as fp:
    lines = fp.readlines()  # read all lines into list
    fp.seek(0) # move file pointer to beginning
    fp.truncate() # truncate file

    fp.write(lines[0]) # write header

    # loop through remaining lines and write back the ones that meet the condition
    for line in lines[1:]:  # skip header
        timestep = int(line.split()[0])

        # only write the lines greater than or equal to start_timestep
        if timestep >= start_timestep:
            fp.write(line)


old_name = path_to_trimmed_outfiles + name_outfile
new_name = path_to_trimmed_outfiles + name_trimmed_outfile

if os.path.isfile(new_name):
    # remove existing file + overwrite it
    os.remove(new_name)

# rename the file to the new trimmed outfile
os.rename(old_name, new_name)