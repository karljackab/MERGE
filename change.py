with open('test_output.csv', 'r') as fr, open('new_test_output.csv', 'w') as fw:
    for row in fr.readlines():
        row = row.strip()
        row = row.replace('"', '')
        fw.write(f'"{row}"\n')