import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Dict

def load_design_from_mat(file_path: str, variable_name: str):
    from scipy.io import loadmat
    data = loadmat(file_path)
    return torch.tensor(data[variable_name].flatten(), dtype=torch.float64)

def plot_results(results: Dict, wavelengths: torch.Tensor, show: bool = True):
    design = results['design'].cpu().numpy()
    actual = results['actual'].cpu().numpy()
    errors = results['errors'].cpu().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].bar(range(len(design)), design, alpha=0.5, label='Design')
    axs[0, 0].bar(range(len(design)), actual[0, :], alpha=0.5, label='Actual')
    axs[0, 0].set_xlabel('Layer index')
    axs[0, 0].set_ylabel('Thickness (nm)')
    axs[0, 0].legend()
    axs[0, 1].bar(range(len(design)), errors[0, :])
    axs[0, 1].set_xlabel('Layer index')
    axs[0, 1].set_ylabel('Error (nm)')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    if show:
        plt.show()
    return fig

def plot_spectra_evolution(
    spectra_history: list,
    layer_index: int,
    chip_index: int = 0,
    wavelengths: Optional[np.ndarray] = None,
    show_measured: bool = True,
    show_theoretical: bool = True,
    cmap='viridis'
):
    entries = [e for e in spectra_history[layer_index] if e.get('chip', 0) == chip_index]
    if not entries:
        print(f"No data for layer {layer_index}, chip {chip_index}")
        return
    entries.sort(key=lambda x: x['time'])
    times = [e['time'] for e in entries]
    if wavelengths is None:
        wavelengths = np.arange(len(entries[0]['measured']))
    else:
        wavelengths = np.array(wavelengths)

    if show_measured:
        meas_matrix = np.array([e['measured'] for e in entries])
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(wavelengths, times, meas_matrix, shading='auto', cmap=cmap)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Layer {layer_index}, Chip {chip_index} – Measured')
        plt.colorbar(im, label='Transmittance (%)')
        plt.show()

    if show_theoretical:
        theo_matrix = np.array([e['theoretical'] for e in entries])
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(wavelengths, times, theo_matrix, shading='auto', cmap=cmap)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Layer {layer_index}, Chip {chip_index} – Theoretical')
        plt.colorbar(im, label='Transmittance (%)')
        plt.show()

def plot_noise_components(
    spectra_history: list,
    layer_index: int,
    chip_index: int = 0,
    step_index: int = -1,
    wavelengths: Optional[np.ndarray] = None
):
    entries = [e for e in spectra_history[layer_index] if e.get('chip', 0) == chip_index]
    if not entries:
        print(f"No data for layer {layer_index}, chip {chip_index}")
        return
    entries.sort(key=lambda x: x['time'])
    if step_index < 0:
        step_index = len(entries) + step_index
    if step_index < 0 or step_index >= len(entries):
        print(f"Invalid step_index {step_index}")
        return
    entry = entries[step_index]
    if 'noise_rand' not in entry:
        print("No noise components saved. Set return_components=True in MeasurementModel.")
        return

    if wavelengths is None:
        wavelengths = np.arange(len(entry['measured']))
    else:
        wavelengths = np.array(wavelengths)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(wavelengths, entry['noise_rand'], 'r-')
    axs[0].set_ylabel('Random noise')
    axs[1].plot(wavelengths, entry['noise_fluc'], 'g-')
    axs[1].set_ylabel('Fluctuation noise')
    axs[2].plot(wavelengths, entry['noise_syst'], 'b-')
    axs[2].set_ylabel('Systematic noise')
    axs[2].set_xlabel('Wavelength (nm)')
    fig.suptitle(f'Layer {layer_index}, Chip {chip_index}, Step {step_index}, t={entry["time"]:.2f}s')
    plt.show()

def animate_spectra_evolution(
    spectra_history: list,
    layer_index: int,
    chip_index: int = 0,
    wavelengths: Optional[np.ndarray] = None,
    show_measured: bool = True,
    show_theoretical: bool = True,
    interval: int = 200
):
    entries = [e for e in spectra_history[layer_index] if e.get('chip', 0) == chip_index]
    if not entries:
        print(f"No data for layer {layer_index}, chip {chip_index}")
        return None
    entries.sort(key=lambda x: x['time'])
    if wavelengths is None:
        wavelengths = np.arange(len(entries[0]['measured']))
    else:
        wavelengths = np.array(wavelengths)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance (%)')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    line_meas, = ax.plot([], [], 'r-', lw=2, label='Measured')
    line_theo, = ax.plot([], [], 'b--', lw=2, label='Theoretical')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    ax.legend()

    def init():
        line_meas.set_data([], [])
        line_theo.set_data([], [])
        time_text.set_text('')
        return line_meas, line_theo, time_text

    def update(frame):
        entry = entries[frame]
        if show_measured:
            line_meas.set_data(wavelengths, entry['measured'])
        else:
            line_meas.set_data([], [])
        if show_theoretical:
            line_theo.set_data(wavelengths, entry['theoretical'])
        else:
            line_theo.set_data([], [])
        time_text.set_text(f'Time = {entry["time"]:.2f} s, d = {entry["actual_thickness"]:.2f} nm')
        return line_meas, line_theo, time_text

    ani = FuncAnimation(fig, update, frames=len(entries), init_func=init, interval=interval, blit=True)
    plt.close(fig)
    return ani