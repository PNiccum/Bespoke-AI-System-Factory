from conf.ini import parse


def get_list(text, section, key):
    """A multi-line setting as a list of its non-empty lines."""
    return [v for v in parse(text)[section][key].split("\n") if v]
