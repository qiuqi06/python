def show(line):
    out, n, tmp, lens = [], 0, "", line.__len__()
    for i in range(lens):
        if line[i] == ',' and n == 0:
            out.append(tmp)
            tmp = ''
        elif line[i] == '"':
            if tmp == '':
                tmp = '@'
                n += 1
            else:
                if tmp[-1] == '#':
                    tmp = tmp[:-1] + '"'
                    n -= 1
                else:
                    tmp = tmp + '#'
                    n += 1
        elif line[i] == ',' and n == 2 and tmp[-1] == '#':
            out.append(tmp[1:-1])
            tmp, n = '', 0
        else:
            tmp = tmp + line[i]
    if tmp[0] == '@':
        out.append(tmp[1:-1])
    else:
        out.append(tmp)
    '\t'.join(out)
    print(out)


with open("data", "r") as file:
    line = file.readline()
    while line:
        show(line)
        line = file.readline()
