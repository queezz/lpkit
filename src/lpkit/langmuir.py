import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt


def find_voltage_peaks(y, height=0.1, distance=0.15e6):
    """Find peaks in a 1D array."""
    peaks, _ = find_peaks(y, height=height, distance=distance)
    return peaks


def average_iv(voltage, current, start=0.0, end=0.0):
    peaks = find_voltage_peaks(-voltage)  # Detect periods
    avg_voltages = []
    avg_currents = []

    for i in range(len(peaks) - 1):
        start_idx, end_idx = peaks[i], peaks[i + 1]
        period_length = end_idx - start_idx

        start_trim = start_idx + int(period_length * start)
        end_trim = start_idx + int(period_length * (1 - end))

        truncated_v = voltage[start_trim:end_trim]
        truncated_i = current[start_trim:end_trim]

        avg_voltages.append(truncated_v)
        avg_currents.append(truncated_i)

    # Ensure uniform length for averaging
    min_length = min(len(p) for p in avg_voltages)

    avg_voltage = np.mean([p[:min_length] for p in avg_voltages], axis=0)
    avg_current = np.mean([p[:min_length] for p in avg_currents], axis=0)

    return avg_voltage, avg_current


def average_iv_backup(voltage, current, start=0.0, end=0.0):
    peaks = find_voltage_peaks(-voltage)
    valid_mask = np.zeros(len(voltage), dtype=bool)

    for i in range(len(peaks) - 1):
        start_idx, end_idx = peaks[i], peaks[i + 1]
        period_length = end_idx - start_idx

        start_trim = start_idx + int(period_length * start)
        end_trim = start_idx + int(period_length * (1 - end))

        valid_mask[start_trim:end_trim] = True

    masked_voltage = voltage[valid_mask]
    masked_current = current[valid_mask]

    period_length = np.min(np.diff(peaks))
    truncated_length = int(period_length * (1 - start - end))
    num_periods = len(masked_voltage) // truncated_length

    avg_voltage = (
        masked_voltage[: num_periods * truncated_length]
        .reshape(num_periods, truncated_length)
        .mean(axis=0)
    )
    avg_current = (
        masked_current[: num_periods * truncated_length]
        .reshape(num_periods, truncated_length)
        .mean(axis=0)
    )

    return avg_voltage, avg_current


def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)  # Zero-phase filtering


def process_lp_data(df, cutoff=1e4, fs=1e6, time_offset=0.33):
    """
    Process Langmuir probe data:
    - Finds voltage peaks
    - Applies a low-pass filter
    - Selects a time window before each peak

    Parameters:
        df (DataFrame): Input data with 'V' (voltage) and 'I' (current).
        cutoff (float): Low-pass filter cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        time_offset (float): Time before each peak to include in mask (s).

    Returns:
        dict: {'time', 'voltage', 'current',
               'voltage_filtered', 'current_filtered', 'peaks', 'mask'}
    """
    voltage = df["V"].values
    current = df["I"].values

    peaks = find_voltage_peaks(-voltage)[1:]

    time = np.arange(len(voltage)) * (1 / fs)

    voltage_filtered = lowpass_filter(voltage, cutoff, fs)
    current_filtered = lowpass_filter(current, cutoff, fs)

    mask = np.zeros_like(time, dtype=bool)
    for peak_idx in peaks:
        start_time = time[peak_idx] - time_offset
        mask |= (time >= start_time) & (time <= time[peak_idx])

    return {
        "time": time,
        "voltage": voltage,
        "current": current,
        "voltage_filtered": voltage_filtered,
        "current_filtered": current_filtered,
        "peaks": peaks,
        "mask": mask,
    }


def plot_lp_data(data, savefile=None, xlim=None):
    """
    Matplotlib plot of raw and filtered Langmuir probe data.

    Parameters:
        data (dict): Processed data from `process_lp_data`
        savefile (str): If provided, saves the figure to this file.
        xlim (tuple): (xmin, xmax) for zooming in on specific time regions.
    """
    plt.figure(figsize=(8, 4))

    # Plot raw signals
    plt.plot(data["time"], data["voltage"], lw=0.5, label="Voltage")
    plt.plot(
        data["time"][data["peaks"]], data["voltage"][data["peaks"]], "x", label="Peaks"
    )
    plt.plot(data["time"], -data["current"], lw=0.5, label="Current")

    plt.plot(
        data["time"],
        -data["current_filtered"],
        c="C1",
        lw=0.5,
        label="Filtered Current",
    )
    for p in data["peaks"]:
        plt.axvline(data["time"][p], c="k", lw=0.5)

    voltage_masked = np.full_like(data["voltage_filtered"], np.nan)
    voltage_masked[data["mask"]] = data["voltage_filtered"][data["mask"]]

    plt.plot(data["time"], voltage_masked, c="k", lw=1, label="Voltage Select")

    plt.xlabel("Time (s)")
    plt.ylabel("Signal (V)")
    plt.legend(loc="upper right")

    if xlim:
        plt.xlim(xlim)

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")

    plt.show()


def average_masked_data(data):
    """
    Extracts and averages time-aligned masked voltage and current segments using peak indices.

    Parameters:
        data (dict): Processed data containing 'time',
                          'voltage_filtered', 'current_filtered', 'mask', and 'peaks'.

    Returns:
        tuple: (avg_time, avg_voltage, avg_current)
    """
    time = data["time"]
    voltage = data["voltage_filtered"]
    current = data["current_filtered"]
    mask = data["mask"]
    peaks = data["peaks"]

    # Extract valid values based on mask
    masked_time = time[mask]
    masked_voltage = voltage[mask]
    masked_current = current[mask]

    # Split data into segments based on peaks
    voltage_segments = [
        voltage[mask & (time >= time[p] - 0.33) & (time <= time[p])] for p in peaks
    ]
    current_segments = [
        current[mask & (time >= time[p] - 0.33) & (time <= time[p])] for p in peaks
    ]
    time_segments = [
        time[mask & (time >= time[p] - 0.33) & (time <= time[p])] for p in peaks
    ]

    # Find the minimum length across all segments to align them
    min_length = min(len(seg) for seg in voltage_segments)

    # Truncate all segments to the same length
    truncated_v = np.array([seg[:min_length] for seg in voltage_segments])
    truncated_i = np.array([seg[:min_length] for seg in current_segments])
    truncated_t = np.array([seg[:min_length] for seg in time_segments])

    # Compute the mean across all peak-aligned segments
    avg_voltage = np.mean(truncated_v, axis=0)
    avg_current = np.mean(truncated_i, axis=0)
    avg_time = np.mean(truncated_t, axis=0)  # Average time reference

    return avg_time, avg_voltage, avg_current


def plot_iv_characteristic(
    i,
    v,
    figsize=(4, 6),
    v_multiplier=1000,
    i_multiplier=100,
    xlim=(-20, 120),
    ylim=(-3, 8),
    savefile=None,
):
    """
    Plots the I-V characteristic curve.

    Parameters:
    i (array-like): Array of current values.
    v (array-like): Array of voltage values.
    figsize (tuple, optional): Size of the figure. Default is (4, 6).
    v_multiplier (float, optional): Multiplier based on the devider. Default is 1000.
    i_multiplier (float, optional): Multiplier based on shunt R. Default is 100.
    xlim (tuple, optional): Limits for the x-axis. Default is (-20, 120).
    ylim (tuple, optional): Limits for the y-axis. Default is (-3, 8).
    savefile (str, optional): Filename to save the plot.
                   If None, the plot is not saved. Default is None.

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    plt.plot(v * v_multiplier, -i * i_multiplier, "k")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (mA)")
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show()


def plot_iv_plotly(
    i,
    v,
    v_multiplier=1000,
    i_multiplier=100,
):
    """
    Plots an I-V curve using Plotly.
    Parameters:
    i (array-like): The current values (in V).
    v (array-like): The voltage values (in V).
    The function scales the voltage by 1000 to convert it to V;
    and scales the current by -100 to convert it to mA.
    Returns:
    None
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=v * v_multiplier,
            y=-i * i_multiplier,
            mode="lines",
            name="IV",
            line=dict(color="#75bcff", width=1),
        )
    )

    fig.update_layout(
        autosize=False,
        width=400,
        height=600,
        title="Langmuir Probe IV Curve",
        xaxis_title="V (V)",
        yaxis_title="I (mA)",
        legend_title="Legend",
        template="plotly_dark",
        plot_bgcolor="#2e2e2e",
        paper_bgcolor="#2e2e2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )

    fig.show()


def plot_raw_data_plotly(data, downdampling=10):
    """
    Plots an I-V curve using Plotly.
    data: dictionary with keys
    ['time','voltage','current','voltage_filtered',
     'current_filtered','peaks','mask']
    """
    fig = go.Figure()

    factor = downdampling
    pt = data["time"][::factor]
    pv = data["voltage"][::factor]
    pi = -data["current"][::factor]

    fig.add_trace(
        go.Scattergl(
            x=pt,
            y=pv,
            mode="lines",
            name="Voltage",
            line=dict(color="#75bcff", width=1),
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=pt, y=pi, mode="lines", name="Current", line=dict(color="red", width=1)
        )
    )

    fig.update_layout(
        title="Raw LP data",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_dark",
        plot_bgcolor="#2e2e2e",
        paper_bgcolor="#2e2e2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )

    fig.show()


def use_pyqtgraph(data):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets
    import numpy as np
    import sys

    time, voltage, current = data

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="Voltage and Current")

    plot = win.addPlot(title="Voltage and Current (Real-Time)")

    plot.setClipToView(True)
    plot.showGrid(x=True, y=True)
    plot.setLabel("left", "Value")
    plot.setLabel("bottom", "Time")

    curve1 = plot.plot(time, voltage, pen=pg.mkPen("#75bcff", width=1), name="Voltage")
    curve2 = plot.plot(time, -current, pen=pg.mkPen("r", width=1), name="Current")

    sys.exit(app.exec_())
