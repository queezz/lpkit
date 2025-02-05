import numpy as np
import struct


def inspect_wvf_header(filename, num_bytes=128):
    """
    Reads and prints the first 'num_bytes' of
    a Yokogawa SL1000 WVF file to analyze its header structure.
    """
    with open(filename, "rb") as file:
        header_bytes = file.read(num_bytes)

    print("Hex Dump of Header:")
    print(" ".join(f"{b:02X}" for b in header_bytes))

    return header_bytes


def find_data_offset(filename, chunk_size=512):
    """
    Scans the WDF file to locate the probable start of waveform data.
    """
    with open(filename, "rb") as file:
        data = file.read(chunk_size)

    offset = next((i for i, b in enumerate(data) if b < 9 or b > 127), None)
    print(f"Possible Data Offset: {offset} bytes")

    return offset


def read_wdf_waveform(filename, data_offset, num_channels=1, data_format="I2"):
    """
    Reads waveform data from a Yokogawa WDF file.

    Parameters:
        filename (str): Path to WDF file.
        data_offset (int): Byte offset where waveform data starts.
        num_channels (int): Number of channels.
        data_format (str): "I2" (16-bit int), "I4" (32-bit int), "F4" (32-bit float).

    Returns:
        np.ndarray: Extracted waveform data.
    """
    data_types = {
        "I2": ("<h", 2),
        "I4": ("<i", 4),
        "F4": ("<f", 4),
    }

    if data_format not in data_types:
        raise ValueError("Unsupported data format")

    fmt_str, num_bytes = data_types[data_format]

    with open(filename, "rb") as file:
        file.seek(data_offset)
        raw_data = file.read()

    total_samples = len(raw_data) // (num_channels * num_bytes)
    y = np.array(struct.unpack(f"{total_samples * num_channels}{fmt_str}", raw_data))
    y = y.reshape((total_samples, num_channels))

    return y


def save_hex_dump(filename, output_file, num_bytes=512):
    """
    Reads the first 'num_bytes' from a file and saves a hex dump to a text file.
    """
    with open(filename, "rb") as f:
        data = f.read(num_bytes)

    with open(output_file, "w") as f:
        f.write(" ".join(f"{b:02X}" for b in data))


def save_repr_format(filename, output_file, num_bytes=512):
    """
    Reads the first 'num_bytes' from a file and saves it in Python's repr() format.
    """
    with open(filename, "rb") as f:
        data = f.read(num_bytes)

    with open(output_file, "w") as f:
        f.write(repr(data))
