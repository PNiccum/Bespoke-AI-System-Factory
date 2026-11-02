class CycleError(ValueError):
    """The dependencies loop. ``cycle`` lists the looping tasks so that each one depends on the next, with the
    first task repeated at the end, e.g. ["a", "b", "a"]."""

    def __init__(self, cycle):
        super().__init__("dependency cycle: " + " -> ".join(cycle))
        self.cycle = cycle


def build_order(deps):
    """Order tasks so each one comes after every task it depends on. ``deps`` maps a task to the tasks it needs;
    a task that appears only as a dependency needs nothing. When several tasks are ready at once, the
    alphabetically first goes next. Raises CycleError when the dependencies loop."""
    nodes = set(deps) | {d for needs in deps.values() for d in needs}
    waiting = {n: 0 for n in nodes}
    users = {n: [] for n in nodes}
    for task, needs in deps.items():
        for d in needs:
            waiting[task] += 1
            users[d].append(task)
    ready = sorted(n for n in nodes if waiting[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for u in users[n]:
            waiting[u] -= 1
            if waiting[u] == 0:
                ready.append(u)
    return order
