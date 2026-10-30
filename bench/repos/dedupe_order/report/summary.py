from report.clean import dedupe


def visited(log):
    """Pages in the order they were first visited."""
    return " > ".join(dedupe(line.split()[1] for line in log))
