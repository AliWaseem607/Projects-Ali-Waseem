import concurrent.futures
import logging
import logging.config
import os
import re
import subprocess
import time
from pathlib import Path

import tyro


def run_radar(
    iteration: int, parameters_file_path: Path, wrfout_file_path: Path, output_dir: Path, quiet: bool = True
):
    cmd = ["bash", "/home/waseem/scripts/crsim_single_runner.sh", "-p", str(parameters_file_path.absolute())]
    if quiet:
        cmd.append("-q")
    cmd.extend([str(output_dir.absolute()), str(wrfout_file_path.absolute()), f"{int(iteration)}"])
    run = subprocess.run(args=cmd, check=True)
    if run.returncode == 1:
        print(f"Iteration: {iteration} failed")


def get_times(wrfout_file_path: Path) -> int:
    ncdump = subprocess.run(
        ["ncdump", "-h", str(wrfout_file_path.absolute())],
        text=True,
        check=True,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    times_res = re.search(r"Time = UNLIMITED ; // \(([\d]+) ", ncdump.stdout)
    assert times_res is not None
    return int(times_res.groups()[0])


def main(
    wrfout_file_path: Path,
    output_dir: Path,
    parameters_file_path: Path,
    quiet: bool = True,
    num_cpus: int | None = None,
):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    ntimes = get_times(wrfout_file_path)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    futures = []

    for timestep in range(280, ntimes + 1, 1):  # index starts at 1
        futures.append(
            executor.submit(
                run_radar,
                iteration=timestep,
                parameters_file_path=parameters_file_path,
                wrfout_file_path=wrfout_file_path,
                output_dir=output_dir,
                quiet=quiet,
            )
        )
        if timestep == 280:
            time.sleep(2)

    executor.shutdown(wait=True, cancel_futures=False)
    for future in futures:
        if future.done() == False:
            print("Something is wrong")
        if future.exception() != None:
            print(future.exception())
            print("exception was raised")


tyro.cli(main)
