import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def find_voltage_peaks(y, height=0.1, distance=0.15e6):
    """Find peaks in a 1D array."""
    peaks, _ = find_peaks(y, height=height, distance=distance)
    return peaks


def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)  # Zero-phase filtering


def apply_mask(mask, *arrays):
    """Apply a mask to multiple numpy arrays."""
    return (arr[mask] for arr in arrays)


# MARK: Data Prep
def process_lp_data(df, metadata, cutoff=1e4, fs=1e6, time_offset=0.33, **kws):
    """
    Process Langmuir probe data by filtering and identifying peaks.
    Parameters:
    df (pandas.DataFrame): DataFrame containing the voltage ('V') and current ('I') data.
    cutoff (float, optional): Cutoff frequency for the lowpass filter. Default is 1e4.
    fs (float, optional): Sampling frequency of the data. Default is 1e6.
    time_offset (float, optional): Time offset to apply when masking around peaks. Default is 0.33.
    **kws: Additional keyword arguments.
        - window (list, optional): Time window [start, end] to crop the data. Default is [time[0], time[-1]].
        - peak_height (float, optional): Minimum height of peaks to detect. Default is 0.1.
        - peak_distance (float, optional): Minimum distance between peaks in seconds.
    Returns:
    dict: A dictionary containing the following keys:
        - 'time' (numpy.ndarray): Time array.
        - 'voltage' (numpy.ndarray): Voltage data.
        - 'current' (numpy.ndarray): Current data.
        - 'voltage_filtered' (numpy.ndarray): Filtered voltage data.
        - 'current_filtered' (numpy.ndarray): Filtered current data.
        - 'peaks' (numpy.ndarray): Indices of detected peaks in the voltage data.
        - 'mask' (numpy.ndarray): Boolean mask array indicating the regions around peaks.
    """

    voltage = df["V"].values
    current = df["I"].values
    fs = 1 / metadata["HResolution"][0]
    time = np.arange(len(voltage)) * (1 / fs)

    window = kws.get("window", [time[0], time[-1]])
    mask = (time > window[0]) & (time < window[1])
    voltage, current, time = apply_mask(mask, voltage, current, time)
    voltage_filtered = lowpass_filter(voltage, cutoff, fs)
    current_filtered = lowpass_filter(current, cutoff, fs)

    peak_height = kws.get("peak_height", 0.1)
    peak_distance = kws.get("peak_distance", 1)
    peak_distance = peak_distance * fs
    peaks = find_voltage_peaks(
        -voltage_filtered, height=peak_height, distance=peak_distance
    )
    peaks = peaks[1:-1]  # Remove first and last peaks

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
        "time_offset": time_offset,
    }


# MARK: Averaging


def make_segments(data):
    """
    Make segments from periodic IV data
    """
    time = data["time"]
    voltage = data["voltage_filtered"]
    current = data["current_filtered"]
    mask = data["mask"]
    peaks = data["peaks"]
    time_offset = data["time_offset"]

    voltage_segments = [
        voltage[mask & (time >= time[p] - time_offset) & (time <= time[p])]
        for p in peaks
    ]
    current_segments = [
        current[mask & (time >= time[p] - time_offset) & (time <= time[p])]
        for p in peaks
    ]

    min_length = min(len(seg) for seg in voltage_segments)
    truncated_v = np.array([seg[:min_length] for seg in voltage_segments])
    truncated_i = np.array([seg[:min_length] for seg in current_segments])
    segments = np.array([truncated_i, truncated_v])

    return segments


def average_segments(segments, num_bins=1000,sigma=2):
    """
    Average segments of IV data
    """
    V_segments = segments[1]
    I_segments = -segments[0]

    # Flatten all segments into a single list
    V_all = np.concatenate(V_segments)
    I_all = np.concatenate(I_segments)

    # Sort by voltage to prepare for binning
    sorted_indices = np.argsort(V_all)
    V_sorted = V_all[sorted_indices]
    I_sorted = I_all[sorted_indices]

    # Adaptive binning
    bins = np.linspace(V_sorted.min(), V_sorted.max(), num_bins)
    bin_indices = np.digitize(V_sorted, bins)

    # Compute median I within each bin (avoiding outliers)
    I_binned = np.array(
        [
            (
                np.median(I_sorted[bin_indices == i])
                if np.any(bin_indices == i)
                else np.nan
            )
            for i in range(1, len(bins))
        ]
    )

    # Bin centers as the final voltage axis
    V_binned = (bins[:-1] + bins[1:]) / 2  # Midpoints of bins

    # Optional: Apply Gaussian smoothing to clean noise    

    I_smooth = gaussian_filter1d(I_binned, sigma=sigma)
    return V_binned, I_smooth

# MARK: Fit LogI
def fit_logi(v,logi,vlim):
    mask = (v > vlim[0]) & (v < vlim[1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=logi, mode='lines', name='IV filtered',
                            line=dict(color='#ebab7a')))
    fig.add_trace(go.Scatter(x=v[mask], y=logi[mask],
                            mode='lines', name='Linear region', 
                            line=dict(width=10, color='rgba(255, 0, 0, 0.5)')
                            ))

    try:
        coeffs = np.polyfit(v[mask], logi[mask], 1)
        vfit = np.linspace(*vlim,50)
        ifit = np.polyval(coeffs, vfit)
        fig.add_trace(go.Scatter(x=vfit, y=ifit, mode='lines', name='Fit'))
        print(f'\033[44mTe = {1/coeffs[0]:.2f} eV\033[0m')
    except TypeError:
        pass

    fig.update_layout(
        title="Getting Te from IV",
        xaxis_title="V (V)",
        yaxis_title="log(I + const)",
        legend_title="Legend",
    )
    plotly_style_dark(fig)

    fig.show()


# MARK: PLOT Raw MPL
def plot_lp_data(
    *,
    time,
    current,
    voltage,
    voltage_filtered,
    current_filtered,
    peaks,
    mask,
    savefile=None,
    xlim=None,
    **kwargs,
):
    """
    Matplotlib plot of raw and filtered Langmuir probe data.

    Parameters:
        data (dict): Processed data from `process_lp_data`
        savefile (str): If provided, saves the figure to this file.
        xlim (tuple): (xmin, xmax) for zooming in on specific time regions.
    """
    plt.figure(figsize=(8, 4))

    plt.plot(time, -current, lw=0.5, label="Current")
    plt.plot(
        time,
        -current_filtered,
        lw=0.5,
        label="Filtered Current",
    )
    plt.plot(time, voltage, lw=0.5, label="Voltage")
    plt.plot(time[peaks], voltage[peaks], "x", label="Peaks")

    for p in peaks:
        plt.axvline(time[p], c="k", lw=0.5)

    voltage_masked = np.full_like(voltage_filtered, np.nan)
    voltage_masked[mask] = voltage_filtered[mask]

    plt.plot(time, voltage_masked, c="k", lw=1, label="Voltage Select")

    plt.xlabel("Time (s)")
    plt.ylabel("Signal (V)")
    plt.legend(loc="upper right")

    if xlim:
        plt.xlim(xlim)

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")

    plt.show()


# MARK: IV MPL
def plot_iv_characteristic(
    i,
    v,
    figsize=(4, 6),
    v_multiplier=1000,
    i_multiplier=100,
    xlim=None,
    ylim=None,
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
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (mA)")
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show()


# MARK: IV Plotly


def plot_iv_plotly(i, v, v_multiplier=1000, i_multiplier=100, **kws):
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

    width = kws.get("width", 400)
    height = kws.get("height", 600)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        title="Langmuir Probe IV Curve",
        xaxis_title="V (V)",
        yaxis_title="I (mA)",
        legend_title="Legend",
    )
    plotly_style_dark(fig)

    fig.show()


# MARK: Raw Plotly
def plot_raw_data_plotly(data, downdampling=10):
    """
    Plots an I-V curve using Plotly.
    data: dictionary with keys
    ['time','voltage','current','voltage_filtered',
     'current_filtered','peaks','mask']
    """
    fig = go.Figure()

    factor = downdampling
    mask = data["mask"]
    voltage_filtered = data["voltage_filtered"]
    t = data["time"]
    v = voltage_filtered
    cur = -data["current_filtered"][::factor]
    peaks = data["peaks"]

    fig.add_trace(
        go.Scattergl(
            x=t[::factor],
            y=cur,
            mode="lines",
            name="Current",
            line=dict(color="red", width=1),
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=t[::factor],
            y=v[::factor],
            mode="lines",
            name="Voltage",
            line=dict(color="#75bcff", width=1),
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=data["time"][peaks],
            y=data["voltage"][peaks],
            mode="markers",
            name="Peaks",
            marker=dict(
                color="#f6fa78",
                size=10,
                symbol="x",
                opacity=1,
            ),
        )
    )
    try:
        indices = np.where(np.diff(mask.astype(int)) != 0)[0]
        for i in range(0, len(indices), 2):
            fig.add_shape(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=t[indices[i]],
                    x1=t[indices[i + 1]],
                    y0=0,
                    y1=1,
                    fillcolor="rgb(188, 237, 124)",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            )
    except IndexError as e:
        print(f"Can't slice properly, {e}")

    fig.update_layout(
        title="Raw LP data",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )
    plotly_style_dark(fig)

    fig.show()


def plotly_style_dark(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#2e2e2e",
        paper_bgcolor="#2e2e2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )


# MARK: pyqtgraph


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
