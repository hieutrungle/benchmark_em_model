import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":

    saved_dict = {}
    time_count = 0
    total_images = 24320
    batch_size_dict = {"batch_size": []}
    num_devices_dict = {"num_devices": []}
    global_batch_size_dict = {"global_batch_size": []}
    time_dict = {"time (s)": []}
    total_images_dict = {"total_images": []}
    throughput_dict = {"throughput (images/s)": []}
    epsilon = 1e-1
    with open(
        "/scratch/user/hieult/research/benchmark_em_model/logs/out.540076-benchmark_no_compile_bs_200_gpu_2-00",
        "r",
    ) as file:
        # read lines
        lines = file.readlines()
        for i, line in enumerate(lines):
            # split line by spaces
            line = line.split()
            # get the values
            if "TRAIN" in line:
                time_count = 0
            if "batch_size" in line:
                global_batch_size = int(line[2])
            elif "num_devices" in line:
                num_devices = int(line[2])
            elif "time:" in line or "time" in line:
                time = float(line[2])
                time_count += 1
            if time_count == 3:
                batch_size = global_batch_size // num_devices
                batch_size_dict["batch_size"].append(batch_size)
                num_devices_dict["num_devices"].append(num_devices)
                global_batch_size_dict["global_batch_size"].append(global_batch_size)
                total_images_dict["total_images"].append(total_images)
                time_dict["time (s)"].append(time)
                throughput = total_images / time if time - epsilon >= 0 else 0
                throughput_dict["throughput (images/s)"].append(throughput)
                time_count = 0

    # from dict to dataframe
    saved_dict = {
        **batch_size_dict,
        **num_devices_dict,
        **global_batch_size_dict,
        **total_images_dict,
        **time_dict,
        **throughput_dict,
    }
    df = pd.DataFrame.from_dict(saved_dict)
    df.to_excel(
        "./gpu_results.xlsx",
        index=True,
    )
