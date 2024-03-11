import librosa
import libfmp.b
import libfmp.c6
import libfmp.b
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from numba import jit
import numpy as np
import pandas as pd
from typing import Tuple

SAMPLES = ['bass-drum', 'click', 'hi-hat']


def fade_signal(signal: np.ndarray,
                fading_duration: float = 0,
                fs: int = 22050) -> np.ndarray:
    """Fade in / out audio signal

    Parameters
    ----------
    signal: np.ndarray
        Signal to be faded

    fs: int, default = 22050
        sampling rate

    fading_duration: float, default = 0
        duration of fade-in and fade-out, in seconds

    Returns
    -------
    normalized_signal: np.ndarray
        Normalized signal
    """
    num_samples = int(fading_duration * fs)

    # if the signal is shorter than twice of the length of the fading duration, multiply signal with sinus half-wave
    if len(signal) < 2 * num_samples:
        signal *= np.sin(np.pi * np.arange(len(signal)) / len(signal))
    else:
        signal[:num_samples] *= np.sin(np.pi * np.arange(num_samples) / fading_duration / 2 / fs)
        signal[-num_samples:] *= np.cos(np.pi * np.arange(num_samples) / fading_duration / 2 / fs)

    return signal


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Max-normalize audio signal

    Parameters
    ----------
    signal: np.ndarray
        Signal to be normalized

    Returns
    -------
    normalized_signal: np.ndarray
        Normalized signal
    """
    normalized_signal = signal / np.max(np.abs(signal))
    return normalized_signal


def warp_sample(sample: np.ndarray,
                reference_pitch: int,
                target_pitch: int,
                target_duration_sec: float,
                gain: float = 1.0,
                fs: int = 22050,
                fading_duration: float = 0.01):
    """This function warps a sample. Given the reference pitch of the sample provided as np.ndarray, the warped version
    of the sample gets pitch-shifted using librosa.effects.pitch_shift(). For the temporal alignment, if the desired
    duration is shorter than the original sample, the sample gets cropped, else if the desired duration is longer of
    the provided sample, the returned signal gets zero-padded at the end.

    Parameters
    ----------
    sample: np.ndarray
        Sample to be warped.

    reference_pitch: int
        Reference pitch for the given sample.

    target_pitch: int
        Target pitch for the warped sample.

    target_duration_sec: float
        Duration, given in seconds, for the returned signal.

    gain: float, default = 1.0
        Gain of the generated tone

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    fading_duration: float, default = 0.01
        Duration of fade in and fade out (to avoid clicks)

    Returns
    -------
    warped_sample: np.ndarray
        Warped sample.
    """
    # Compute pitch difference
    pitch_steps = target_pitch - reference_pitch

    # Apply pitch-shifting to original sample
    pitch_shifted_sample = librosa.effects.pitch_shift(y=sample,
                                                       sr=fs,
                                                       n_steps=pitch_steps)

    # Case: target duration is shorter than sample -> cropping
    if int(target_duration_sec * fs) <= len(sample):

        warped_sample = pitch_shifted_sample[:int(target_duration_sec * fs)]

    # Case: target duration is longer than sample -> zero-filling
    else:

        warped_sample = np.zeros(int(target_duration_sec * fs))
        warped_sample[:len(sample)] = sample

    warped_sample = fade_signal(signal=warped_sample, fs=fs, fading_duration=fading_duration)

    warped_sample *= gain

    return warped_sample


def pitch_to_frequency(pitch: int,
                       reference_pitch: int = 69,
                       tuning_frequency: float = 440.0) -> float:
    """Calculates the corresponding frequency for a given pitch.

    Parameters
    ----------
    pitch: int
        Pitch to calculate frequency for.

    reference_pitch: int, default = 69
        Reference pitch for calculation.

    tuning_frequency: float, default = 440.0
        Tuning frequency for calculation, in Hertz.

    Returns
    -------
    frequency: float
        Calculated frequency for given pitch, in Hertz.
    """

    frequency = tuning_frequency * 2 ** ((pitch - reference_pitch) / 12)

    return frequency


# Taken from FMP Notebooks, https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempoBeat.html
def plot_sonify_novelty_beats(fn_wav, fn_ann, title=''):
    ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)
    df = pd.read_csv(fn_ann, sep=';', keep_default_na=False, header=None)
    Fs = 22050
    x, _ = librosa.load(fn_wav, sr=Fs)
    x_duration = len(x) / Fs
    nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=256, gamma=1, M=10, norm=1)
    figsize = (8, 1.5)
    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', figsize=figsize,
                                         title=title)
    libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                                  nontime_axis=True, time_min=0, time_max=x_duration)
    plt.show()

    return fig, ax


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().rename(columns=str.lower)

    if 'duration' not in df.columns:
        try:
            df['duration'] = df['end'] - df['start']
        except ValueError:
            print('Input DataFrame must have start and duration/end columns.')
    else:
        try:
            df['end'] = df['start'] + df['duration']
        except ValueError:
            print('Input DataFrame must have start and duration/end columns.')

    return df


def mix_sonification_and_original(sonification: np.ndarray,
                                  original_audio: np.ndarray,
                                  gain_lin_sonification: float = 1.0,
                                  gain_lin_original_audio: float = 1.0,
                                  panning: float = 1.0,
                                  duration: int = None):
    """This function takes a sonification and an original_audio and mixes it to stereo

    Parameters
    ----------
    sonification: np.ndarray
        Sonification

    original_audio: np.ndarray
        Original audio

    gain_lin_sonification: float, default = 1.0
        linear gain for sonification

    gain_lin_original_audio: float, default = 1.0
        linear gain for original audio

    panning: float, default = 1.0
        Controls the panning of the mixed output
            panning = 1.0 means original audio on left and sonification on right channel
            panning = 0.5 means same amount of both signals on both channels.
            panning = 0.0 means sonification on left and original audio on right channel

    duration: int, default = None
        Duration of the output waveform, given in samples.

    Returns
    -------
    mixed_audio : np.ndarray
        Mix of the signals
    """
    assert 0.0 <= panning <= 1.0, f'Panning must a value between 0.0 and 1.0.'
    if duration is None:
        num_samples = len(original_audio)

    else:
        num_samples = duration

        if len(original_audio) < num_samples:
            original_audio = np.append(original_audio, np.zeros(num_samples - len(original_audio)))

        else:
            original_audio = original_audio[:num_samples]

    if len(sonification) < num_samples:
        sonification = np.append(sonification, np.zeros(num_samples - len(sonification)))

    else:
        sonification = sonification[:num_samples]

    # Perform RMS normalization
    # Calculate the RMS amplitude of each signal
    rms_signal1 = np.sqrt(np.mean(np.square(original_audio)))
    rms_signal2 = np.sqrt(np.mean(np.square(sonification)))

    # Normalize the signals to have the same RMS amplitude
    normalized_signal1 = original_audio * (rms_signal2 / rms_signal1)
    normalized_signal2 = sonification * (rms_signal1 / rms_signal2)

    stereo_audio = np.column_stack(
        (panning * (gain_lin_original_audio * normalized_signal1) +
         (1 - panning) * (gain_lin_sonification * normalized_signal2),
         panning * (gain_lin_sonification * normalized_signal2) +
         (1 - panning) * (gain_lin_original_audio * normalized_signal1))).T

    return stereo_audio


@jit(nopython=True)
def smooth_weights(weights: np.ndarray,
                   fading_samples: int = 0) -> np.ndarray:
    """Weight smoothing

    Parameters
    ----------
    weights: np.ndarray
        Input weights

    fading_samples: int
        Number of samples for fade-in/out.

    Returns
    -------
    weights_smoothed: np.ndarray
        Smoothed weights
    """

    weights_smoothed = weights.copy()

    for i in range(1, len(weights) - 1):
        if weights[i] != weights[i - 1]:
            amplitude = (np.abs(weights[i - 1] - weights[i])) / 2

            x = np.linspace(-1 * (np.pi / 2), np.pi / 2, fading_samples) * -1 * np.sign(
                weights[i - 1] - weights[i])
            y = amplitude * np.sin(x) + (weights[i - 1] + weights[i]) / 2
            start_idx = i - int(fading_samples / 2)
            end_idx = start_idx + len(y)

            if start_idx >= 0 and end_idx < len(weights_smoothed):
                weights_smoothed[start_idx:end_idx] = y

    return weights_smoothed


def visualize_pianoroll(pianoroll_df: pd.DataFrame,
                        xlabel: str = 'Time (seconds)',
                        ylabel: str = 'Pitch',
                        title: str = None,
                        colors: str = 'FMP_1',
                        velocity_alpha: bool = False,
                        figsize : Tuple[float, float] = (12, 4),
                        ax: matplotlib.axes = None,
                        dpi: int = 72) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Visualization function for piano-roll representations, given in a pd.DataFrame format

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe containing pitch-event information.

    xlabel: str, default = 'Time (seconds)'
        Label text for the x-axis.

    ylabel: str, default = 'Pitch'
        Label text for the y-axis.

    title: str, default = None
        Title of the figure.

    colors: str, default = 'FMP_1'
        Colormap, for the default colormap see https://github.com/meinardmueller/libfmp.

    velocity_alpha: bool = False
        Set True to weight the visualized rectangular regions for each pitch based on their velocity value.

    figsize: Tuple[float, float], default: [12, 4])

    ax: matplotlib.axes.Axes
         Axes object

    dpi: int
        Resolution of the figure.


    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure instance

    ax: matplotlib.axes.Axes
         Axes object
    """

    pianoroll_df = format_df(pianoroll_df)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    labels_set = sorted(pianoroll_df['label'].unique())
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)

    pitch_min = pianoroll_df['pitch'].min()
    pitch_max = pianoroll_df['pitch'].max()
    time_min = pianoroll_df['start'].min()
    time_max = pianoroll_df['end'].max()

    for i, r in pianoroll_df.iterrows():
        velocity = None if not velocity_alpha else r['velocity']
        rect = patches.Rectangle((r['start'], r['pitch'] - 0.5), r['duration'], 1, linewidth=1,
                                 edgecolor='k', facecolor=colors[r['label']], alpha=velocity)
        ax.add_patch(rect)

    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label=title)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    if fig is not None:
        plt.tight_layout()

    return fig, ax
