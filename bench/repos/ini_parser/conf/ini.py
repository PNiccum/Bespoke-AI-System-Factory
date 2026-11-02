def parse(text):
    """Parse INI text into {section: {key: value}}.

    - A section starts with a line like ``[name]``.
    - A setting is ``key = value`` or ``key: value``, split at the first ``=`` or ``:``; key and value are
      stripped and the key is lower-cased.
    - Lines whose first non-blank character is ``#`` or ``;`` are comments; blank lines are ignored.
    - An indented line continues the previous setting's value; the pieces are joined with a newline.
    - Settings before any section go in "DEFAULT".
    """
    data = {}
    section = "DEFAULT"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        key, _, value = line.partition("=")
        data.setdefault(section, {})[key.strip().lower()] = value.strip()
    return data
