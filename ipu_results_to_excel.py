import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":

    saved_dict = {}
    time_count = 0
    total_images = 1216
    num_replicas_dict = {"num_replicas": []}
    batch_size_dict = {"batch_size": []}
    device_iterations_dict = {"device_iterations": []}
    num_ipus_pipeline_dict = {"num_ipus_pipeline": []}
    total_ipus_dict = {"total_ipus": []}
    gradient_accumulation_dict = {"gradient_accumulation": []}
    global_batch_size_dict = {"global_batch_size": []}
    time_dict = {"time (s)": []}
    total_images_dict = {"total_images": []}
    throughput_dict = {"throughput (images/s)": []}
    epsilon = 1e-1
    with open("ipu_results_rerun.txt", "r") as file:
        # read lines
        lines = file.readlines()
        for i, line in enumerate(lines):
            # split line by spaces
            line = line.split()
            # print(f"line {i}: {line}")
            # get the values
            if "TRAIN" in line:
                time_count = 0
            if "batch_size" in line:
                batch_size = int(line[2])
            elif "replication_factor" in line:
                num_replicas = int(line[2])
            elif "device_iterations" in line:
                device_iterations = int(line[2])
            elif "num_ipus" in line:
                num_ipus = int(line[2])
            elif "gradient_accumulation" in line:
                gradient_accumulation = int(line[2])
            elif "time:" in line or "time" in line:
                time = float(line[2])
                time_count += 1
            if time_count == 3:
                num_replicas_dict["num_replicas"].append(num_replicas)
                device_iterations_dict["device_iterations"].append(device_iterations)
                total_ipus_dict["total_ipus"].append(num_ipus * num_replicas)
                batch_size_dict["batch_size"].append(batch_size)
                num_ipus_pipeline_dict["num_ipus_pipeline"].append(num_ipus)
                gradient_accumulation_dict["gradient_accumulation"].append(
                    gradient_accumulation
                )
                global_batch_size_dict["global_batch_size"].append(
                    batch_size
                    * num_replicas
                    * device_iterations
                    * gradient_accumulation
                )
                total_images_dict["total_images"].append(total_images)
                time_dict["time (s)"].append(time)
                throughput = total_images / time if time - epsilon >= 0 else 0
                throughput_dict["throughput (images/s)"].append(throughput)
                time_count = 0

    # from dict to dataframe
    saved_dict = {
        **num_replicas_dict,
        **num_ipus_pipeline_dict,
        **total_ipus_dict,
        **batch_size_dict,
        **device_iterations_dict,
        **gradient_accumulation_dict,
        **global_batch_size_dict,
        **total_images_dict,
        **time_dict,
        **throughput_dict,
    }
    df = pd.DataFrame.from_dict(saved_dict)
    df.to_excel("tmp_ipu_results_rerun.xlsx", index=True)
