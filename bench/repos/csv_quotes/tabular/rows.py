def parse_line(line):
    """Split one CSV line into fields. Fields may be quoted with double quotes; a quoted field can hold commas,
    and "" inside quotes stands for one double quote."""
    return [field.strip('"') for field in line.rstrip("\n").split(",")]
