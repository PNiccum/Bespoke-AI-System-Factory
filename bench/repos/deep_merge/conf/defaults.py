from conf.merge import merge

DEFAULTS = {"db": {"host": "localhost", "port": 5432}, "debug": False}


def load(user):
    return merge(DEFAULTS, user)
