import pandas as pd
import numpy as np


def sanitize_var_name(name):
    """Sanitize variable names for NetCDF: remove invalid characters and enforce naming rules."""
    name = name.strip().strip('"')
    return name


def parse_yokogawa_csv(csv_file):
    """Reads a Yokogawa CSV file and extracts metadata + waveform data."""
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = {}
    data_start = None

    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        key = parts[0].strip('"')

        if key == "":
            data_start = i
            break

        values = [v.strip() for v in parts[1:]]
        if key in [
            "TraceName",
            "BlockSize",
            "Date",
            "Time",
            "VUnit",
            "HResolution",
            "HOffset",
            "HUnit",
        ]:
            values.pop()

        values = [sanitize_var_name(v) for v in values]
        metadata[key] = values if len(values) > 1 else values[0]

    return data_start, metadata


def read_yokogawa_csv(csv_file, names=None):
    data_start, metadata = parse_yokogawa_csv(csv_file)
    if names is not None:
        metadata["TraceName"] = names
    df = pd.read_csv(
        csv_file,
        skiprows=data_start,
        names=metadata["TraceName"],
        usecols=range(1, len(metadata["TraceName"]) + 1),
    )
    return df, metadata


def save_npz_file(df, metadata, npz_file):
    np.savez_compressed(
        npz_file, **{col: df[col].values for col in df.columns}, metadata=metadata
    )


def read_npz_file(npz_file):
    """Reads a NumPy .npz file and reconstructs metadata + waveform
    data into a pandas DataFrame."""
    with np.load(npz_file, allow_pickle=True) as data:
        metadata = data["metadata"].item()
        df = pd.DataFrame({col: data[col] for col in metadata["TraceName"]})

    for item in ["HResolution", "HOffset"]:
        metadata[item] = [float(i) for i in metadata[item]]

    metadata["BlockSize"] = [int(i) for i in metadata["BlockSize"]]

    return df, metadata
