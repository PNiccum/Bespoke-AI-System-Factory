"""python -m factory.mcp [--root PATH ...]: serve the factory's tools over stdio."""

import argparse

from factory.mcp.server import build_server

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m factory.mcp")
    ap.add_argument("--root", action="append", default=[], help="a folder the tools may read (repeatable); "
                    "default FACTORY_MCP_ROOTS, else the current directory")
    build_server(ap.parse_args().root or None).run("stdio")
