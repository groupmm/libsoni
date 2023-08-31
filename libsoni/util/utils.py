import numpy as np
import pandas as pd
import librosa
from matplotlib import pyplot as plt
from matplotlib import patches
import os
import json
import libfmp.b
import libfmp.c6
from typing import Dict
from IPython import display as ipd

SAMPLES = ['bass-drum', 'click', 'hi-hat']
try:
    PRESETS = json.load(open(os.path.join('libsoni', 'util', 'presets.json')))
except:
    # TODO: Clean up this mess, this is a workaround for the Sphinx documentation
    PRESETS = json.load(open(os.path.join('..', '..', 'libsoni', 'libsoni', 'util', 'presets.json')))


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize audio signal
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
                fs=22050):
    """TODO: Docstrings
    Parameters
    ----------
    sample
    reference_pitch
    target_pitch
    target_duration_sec
    fs

    Returns
    -------

    """
    pitch_steps = target_pitch - reference_pitch

    pitch_shifted_sample = librosa.effects.pitch_shift(y=sample,
                                                       sr=fs,
                                                       n_steps=pitch_steps)

    rate = len(sample) / int(target_duration_sec * fs)

    time_streched_sample = librosa.effects.time_stretch(y=pitch_shifted_sample,
                                                        rate=rate)

    return time_streched_sample


def get_preset(preset_name: str = None) -> Dict:
    """Get preset parameters from presets.json

    Parameters
    ----------
    preset_name: str, default: None
        Name of preset, e.g., violin
    Returns
    -------
    dictionary of partials, envelope, etc.
    """
    if preset_name not in PRESETS:
        raise ValueError(f'Preset {preset_name} not valid! Choose among {PRESETS.keys()}')
    return PRESETS[preset_name]


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
