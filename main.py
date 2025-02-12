#!/usr/bin/env python3
# LLM Bench - a tool for benchmarking runtime of LLM
# Copyright (C) 2025 Jonathan Trenesaygues <jonathan.tremesaygues@slaanesh.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from argparse import ArgumentParser, Namespace
from itertools import chain
from os import uname
from pathlib import Path
from time import monotonic

import ollama
import plotly.express as px
import polars as pl
import tqdm

MODELS = list(
    chain.from_iterable(
        [f"{model}:{tag}" for tag in tag]
        for (model, tag) in {
            "qwen2.5": ("0.5b", "1.5b", "3b", "7b", "14b", "32b"),
        }.items()
    )
)
ITERS_COUNT = 5


def cmd_bench(args: Namespace) -> None:
    if args.pull:
        for model in tqdm.tqdm(args.models, "Pulling models"):
            ollama.pull(model)

    messages = [
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ]

    dfs = []
    for model in tqdm.tqdm(args.models, "Running models"):
        durations = []
        for _ in tqdm.tqdm(range(args.iters), model):
            clock_start = monotonic()
            ollama.chat(model=model, messages=messages)
            clock_end = monotonic()
            duration = clock_end - clock_start
            durations.append(duration)

        min_duration = min(durations)

        df_model = pl.DataFrame(
            {"host": [args.name], "model": [model], "duration": [min_duration]}
        )
        dfs.append(df_model)

    df = pl.concat(dfs)
    df.write_csv(f"results_{args.name}.csv")


def cmd_aggregate_results(args: Namespace) -> None:
    df = pl.concat([pl.read_csv(run) for run in args.runs])
    df.write_csv("results.csv")


def cmd_plot_results(args: Namespace) -> None:
    df = pl.read_csv(args.results)
    fig = px.line(df, x="model", y="duration", color="host")
    fig.show()


def main() -> None:
    arg_parser = ArgumentParser()

    sub_parsers = arg_parser.add_subparsers()

    # Benchmark subcommand
    cmd_bench_parser = sub_parsers.add_parser("bench", help="Run the benchmark")
    cmd_bench_parser.add_argument(
        "--pull", "-p", action="store_true", help="Pull models from registry"
    )
    cmd_bench_parser.add_argument(
        "--models", "-m", nargs="+", default=MODELS, help="Models to use"
    )
    cmd_bench_parser.add_argument(
        "--iters", "-i", type=int, default=ITERS_COUNT, help="Iterations per model"
    )
    cmd_bench_parser.add_argument(
        "--name",
        "-n",
        default=uname().nodename.split(".", 1)[0],
        help="Name of the run",
    )
    cmd_bench_parser.set_defaults(cmd=cmd_bench)

    # Aggregate subcommand
    cmd_agg_parser = sub_parsers.add_parser(
        "agg", help="Aggregate results from different runs"
    )
    cmd_agg_parser.add_argument("runs", nargs="+", type=Path, help="results files")
    cmd_agg_parser.set_defaults(cmd=cmd_aggregate_results)

    # Plot subcommand
    cmd_plot_parser = sub_parsers.add_parser("plot", help="Plot the results")
    cmd_plot_parser.add_argument(
        "results", nargs="?", type=Path, default="results.csv", help="results file"
    )
    cmd_plot_parser.set_defaults(cmd=cmd_plot_results)

    args = arg_parser.parse_args()
    try:
        cmd = args.cmd
    except AttributeError:
        arg_parser.print_help()
        return
    cmd(args)


if __name__ == "__main__":
    main()
