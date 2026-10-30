from tabular.rows import parse_line


def column(text, name):
    lines = text.splitlines()
    header = parse_line(lines[0])
    i = header.index(name)
    return [parse_line(line)[i] for line in lines[1:]]
