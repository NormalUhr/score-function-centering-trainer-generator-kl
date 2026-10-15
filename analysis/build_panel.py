"""Build the results page from a runs directory: the experiments-at-a-glance
table, the payload timing table and the reward charts, as one self-contained
HTML file.

    python analysis/build_panel.py results/runs results/panel.html
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import at_a_glance  # noqa: E402
import progress_page  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", help="directory of run directories (results/runs, or outputs/ of your own runs)")
    parser.add_argument("out", help="where to write the page")
    parser.add_argument("--title", default="SFC+KL: experiments")
    args = parser.parse_args()
    built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    glance = at_a_glance.build(args.runs, f"Built from {args.runs} on {built}")
    Path(args.out).write_text(progress_page.build(args.runs, glance, updated=built, title=args.title))
    print(args.out)


if __name__ == "__main__":
    main()
