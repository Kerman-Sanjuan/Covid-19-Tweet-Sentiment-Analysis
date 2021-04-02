import sys

raw_file = open(sys.argv[1], 'r', encoding="cp1252")
raw_data_lines = raw_file.readlines()
with open(sys.argv[2], 'w') as f:
    f.write(",text,class\n")
    for idx, val in enumerate(raw_data_lines):
        f.write(str(idx) + ',' + val.strip() + ',?\n')
