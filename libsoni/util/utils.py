import numpy as np
import pandas as pd
import librosa
import libfmp.b
import libfmp.c6
import libfmp.b
from matplotlib import pyplot as plt
from matplotlib import patches

SAMPLES = ['bass-drum', 'click', 'hi-hat']


def fade_signal(signal: np.ndarray = None,
                fading_duration: float = 0.01,
                fs: int = 22050) -> np.ndarray:
    """Fade in / out audio signal
    Parameters
    ----------
    signal: np.ndarray, default = None
        Signal to be faded
    fs: int, default = 22050
        sampling rate
    fading_duration: float, default = 0
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
                fs=22050,
                fading_duration: float = 0.01):
    """This function warps a sample. Given the reference pitch of the sample provided as np.ndarray,
    the warped version of the sample gets pitch-shifted using librosa.effects.pitch_shift().
    For the temporal alignment, if the desired duration is shorter than the original sample, the sample gets cropped,
    else if the desired duration is longer of the provided sample, the returned signal gets zero-padded at the end.

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
    fs: int, default = 22050
        Sampling rate, in samples per seconds.
    fading_sec: float, default = 0.01
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
    """Calculates frequency for pitch.

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
        sonification
    original_audio: np.ndarray
        original_audio
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


def smooth_weights(weights: np.ndarray, fading_samples: int = 0):
    weights_smoothed = np.copy(weights)

    for i in range(1, len(weights) - 1):
        if weights[i] != weights[i - 1]:
            amplitude = (np.abs(weights[i - 1] - weights[i])) / 2

            x = np.linspace(-1 * (np.pi / 2), np.pi / 2, fading_samples) * -1 * np.sign(
                weights[i - 1] - weights[i])

            y = amplitude * np.sin(x) + (weights[i - 1] + weights[i]) / 2
            # print(len(weights_smoothed[i - int(fading_samples / 2):i - int(fading_samples / 2) + len(y)] ))

            weights_smoothed[i - int(fading_samples / 2):i - int(fading_samples / 2) + len(y)] = y
    return weights_smoothed


def envelope_signal(signal: np.ndarray, attack_time: float = 0, decay_time: float = 0, sustain_level: float = 0,
                    release_time: float = 0, fs=44100):
    """
    Envelopes a given signal. If the length of the signal is too short regarding the specified ADSR parameters, the returned signal is zero.
    Parameters
    ----------
    signal : array-like
        signal to envelope
    Returns
    ----------
    enveloped_signal: array-like
        enveloped signal
    """
    if attack_time <= 0 or decay_time <= 0 or release_time <= 0:
        return np.zeros(len(signal))

    # compute lengths of attack, decay, sustain and release section
    attack_samples = int(np.floor(attack_time * fs))
    decay_samples = int(np.floor(decay_time * fs))
    release_samples = int(np.floor(release_time * fs))
    sustain_samples = int(len(signal) - (attack_samples + decay_samples + release_samples))

    # check if signal is at least as long as attack, decay and release section
    if len(signal) < (attack_samples + decay_samples + release_samples):
        return np.zeros(len(signal))

    # compute attack section of envelope
    attack_func = np.exp(np.linspace(0, 1, int(np.floor(attack_time * fs)))) - 1
    attack_func = attack_func / np.max(np.flip(attack_func))

    # compute decay section of envelope
    decay_func = np.exp(np.linspace(0, 1, decay_samples)) - 1
    decay_func = np.flip(sustain_level + (1 - sustain_level) * (decay_func / np.max(decay_func)))

    # compute sustain section of envelope
    sustain_func = sustain_level * np.ones(sustain_samples)

    # compute release section of envelope
    release_func = np.exp(np.linspace(0, 1, release_samples)) - 1
    release_func = np.flip(sustain_level * (release_func / np.max(release_func)))

    # concatenate sections and envelope signal
    enveloped_signal = signal * np.concatenate([attack_func, decay_func, sustain_func, release_func])

    return enveloped_signal


def visualize_pianoroll(df: pd.DataFrame,
                        xlabel: str = 'Time (seconds)',
                        ylabel: str = 'Pitch',
                        title: str = None,
                        colors: str = 'FMP_1',
                        velocity_alpha=False,
                        figsize=(12, 4),
                        ax=None,
                        dpi=72):
    # TODO: dtypes
    """Plot a pianoroll visualization, inspired from FMP Notebook C1/C1S2_CSV.ipynb

    Parameters
    ----------
        df:

    Args:
        score: List of note events
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = 'Pitch')
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications,
            4. dict that assigns labels  to colors (Default value = 'FMP_1')
        velocity_alpha: Use the velocity value for the alpha value of the corresponding rectangle
            (Default value = False)
        figsize: Width, height in inches (Default value = (12)
        ax: The Axes instance to plot on (Default value = None)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes
    """
    df = format_df(df)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    labels_set = sorted(df['label'].unique())
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)

    pitch_min = df['pitch'].min()
    pitch_max = df['pitch'].max()
    time_min = df['start'].min()
    time_max = df['end'].max()

    for i, r in df.iterrows():
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
