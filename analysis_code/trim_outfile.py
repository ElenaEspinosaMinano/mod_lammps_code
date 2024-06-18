### Trim transient part of outfile
"""
name_outfile = pass
name_trimmed_outfile = pass
start_timestep = 1000000 # steady state starts from timestep 1x10^6

# specify paths
path_to_outfiles = '../outfiles/'
path_to_trimmed_outfiles = '../outfiles/trimmed_outfiles/'

n_lines = lines_in_file(path_to_outfiles + name_outfile) # no of lines in file

# open the input file
file_in = open(path_to_outfiles + name_outfile, 'r')

# open the output file and print a header
file_out = open(path_to_trimmed_outfiles + name_trimmed_outfile, 'w')  
file_out.write("# Timesteps, No of clusters, Mean cluster size, Size largest cluster, No clusters size 1, No proteins bound to poly\n")



with open(r"E:\demos\files\sample.txt", 'r+') as fp:
    # read an store all lines into list
    lines = fp.readlines()
    # move file pointer to the beginning of a file
    fp.seek(0)
    # truncate the file
    fp.truncate()

    # start writing lines
    # iterate line and line number
    for number, line in enumerate(lines):
        # delete line number 5 and 8
        # note: list index start from 0
        if number not in [4, 7]:
            fp.write(line)





# close the files
file_in.close()
file_out.close()
"""


"""
input_filename = 'data.txt'  # Replace with your input file name
start_timestep = 100000  # Replace with the timestep from which you want to keep the data

with open(input_filename, 'r+') as fp:
    lines = fp.readlines()  # Read all lines into a list
    fp.seek(0)  # Move file pointer to the beginning
    fp.truncate()  # Truncate the file

    # Write the header
    fp.write(lines[0])

    # Iterate through the remaining lines and write back the ones that meet the condition
    for line in lines[1:]:  # Skip the header
        timestep = int(line.split()[0])
        if timestep >= start_timestep:
            fp.write(line)

print(f'Trimmed data has been written to {input_filename}')
"""