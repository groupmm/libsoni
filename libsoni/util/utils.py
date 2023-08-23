import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import os
import json
from typing import Dict

SAMPLES = ['bass-drum', 'click', 'hi-hat']

PRESETS = json.load(open(os.path.join('libsoni', 'util', 'presets.json')))


def normalize(signal: np.ndarray) -> np.ndarray:
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


def generate_click(pitch: int = 69,
                   amplitude: float = 1.0,
                   duration: float = 0.2,
                   fs: int = 22050,
                   tuning_frequency: float = 440.0) -> np.ndarray:
    """Returns a click signal.

    Parameters
    ----------
    pitch : int, default = 69
        Pitch for colored click.
    amplitude : float, default = 1.0
        Amplitude of click signal.
    duration : float, default = 0.2
        Duration of click signal.
    fs : int, default = 22050
        Sampling rate.
    tuning_frequency : int, default = 440
        Tuning frequency.
    Returns
    -------
    click : np.ndarray
        Click signal.
    """
    click_freq = tuning_frequency * 2 ** ((pitch - 69) / 12)
    angular_freq = 2 * np.pi * click_freq / float(fs)
    click = np.logspace(0, -10, num=int(fs * duration), base=2.0)
    click *= np.sin(angular_freq * np.arange(len(click)))
    click *= amplitude
    return click


def mix_sonification_and_original(sonification: np.ndarray,
                                  original_audio: np.ndarray,
                                  gain_lin_sonification: float = 1.0,
                                  gain_lin_original_audio: float = 1.0,
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
    duration: int, default = None
        Duration of the output waveform, given in samples.
    Returns
    -------
    stereo_audio : np.ndarray
        Stereo mix of the signals
    """
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
        (gain_lin_original_audio * normalized_signal1, gain_lin_sonification * normalized_signal2)).T

    return stereo_audio


def generate_shepard_tone(pitch_class: int = 0,
                          filter: bool = False,
                          f_center: float = 440.0,
                          octave_cutoff: int = 1,
                          amplitude: float = 1.0,
                          duration: float = 1.0,
                          fs: int = 44100,
                          f_tuning: float = 440,
                          fade_dur: float = 0.01,
                          ) -> np.ndarray:
    """Generate shepard tone

        Args:
            pitch_class: int (default: 0)
                pitch class of the synthesized tone
            filter: bool (default: False)
                decides, if shepard tones are filtered or not
            f_center : float (default: 440.0)
                center_frequency in Hertz for bell-shaped filter
            octave_cutoff: int (default: 1)
                determines, at which multiple of f_center, the harmonics get attenuated by 2.
            amplitude: float (default: 1.0)
                amplitude of resulting signal
            duration: float (default: 1.0)
                duration (in seconds)
            fs: int (default: 44100)
                sampling rate in Samples/second
            f_tuning: float (default: 440.0)
                tuning frequency (in Hz)
            fade_dur: float (default: 0.01)
                duration (in seconds) of fade in and fade out (to avoid clicks)

        Returns:
            y: synthesized tone
    """
    assert 0 <= pitch_class <= 11, "pitch class out of range"

    N = int(duration * fs)
    t = np.arange(N) / fs
    freqs = f_tuning * 2 ** ((pitch_class + np.arange(11) * 12 - 69) / 12)
    y = np.zeros(N)

    if duration < fade_dur * 2:
        return y
    if filter:
        f_log = 2 * np.logspace(1, 4, 20000)
        f_lin = np.linspace(20, 20000, 20000)
        f_center_lin = np.argmin(np.abs(f_log - f_center))
        weights = np.exp(- (f_lin - f_center_lin) ** 2 / (1.4427 * ((octave_cutoff * 2) * 1000) ** 2))

        for freq in freqs:
            y += weights[np.argmin(np.abs(f_log - freq))] * np.sin(2 * np.pi * freq * t)

    else:
        for freq in freqs:
            y += np.sin(2 * np.pi * freq * t)

    fade_samples = int(fade_dur * fs)

    y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
    y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amplitude * (y / np.max(y))
    return y


def generate_tone_additive_synthesis(pitch: int = 69,
                                     partials: np.ndarray = np.array([1]),
                                     partials_amplitudes: np.ndarray = None,
                                     partials_phase_offsets: np.ndarray = None,
                                     gain: float = 1.0,
                                     duration_sec: float = 1.0,
                                     fs: int = 22050,
                                     f_tuning: float = 440,
                                     fading_sec: float = 0.01):
    """Generates signal using additive synthesis.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the synthesized tone.
    partials: np.ndarray (default = [1])
        An array containing the desired partials of the fundamental frequencies for sonification.
            An array [1] leads to sonification with only the fundamental frequency core,
            while an array [1,2] causes sonification with the fundamental frequency and twice the fundamental frequency.
    partials_amplitudes: np.ndarray, default = [1]
        Array containing the amplitudes for partials.
            An array [1,0.5] causes the sinusoid with frequency core to have amplitude 1,
            while the sinusoid with frequency 2*core has amplitude 0.5.
    partials_phase_offsets: np.ndarray, default = [0]
        Array containing the phase offsets for partials.
    gain: float, default = 1.0
        Gain for generated signal.
    duration_sec: float, default = 1.0
        Duration of generated signal, in seconds.
    fs: int, default = 22050
        Sampling rate, in samples per seconds,
    f_tuning: float, default = 440.0
        Tuning frequency, given in Hertz.
    fading_sec: float, default = 0.01
        Duration of fade in and fade out (to avoid clicks)

    Returns
    -------
    generated_tone: np.ndarray
        Generated signal
    """
    if partials_amplitudes is None:
        partials_amplitudes = np.ones(len(partials))

    if partials_phase_offsets is None:
        partials_phase_offsets = np.zeros(len(partials))

    assert len(partials) == len(partials_amplitudes) == len(partials_phase_offsets),\
        'Partials, Partials_amplitudes and Partials_phase_offsets must be of equal length.'

    num_samples = int(duration_sec * fs)

    t = np.arange(num_samples) / fs

    generated_tone = np.zeros(len(t))

    pitch_frequency = f_tuning * 2 ** ((pitch - 69) / 12)

    if num_samples < 2 * int(fading_sec * fs):
        fading_sec = 0

    for partial, partial_amplitude, partials_phase_offset in zip(partials, partials_amplitudes, partials_phase_offsets):
        generated_tone += partial_amplitude * np.sin(2 * np.pi * pitch_frequency * partial * t + partials_phase_offset)

    if fading_sec != 0:
        fading_samples = int(fading_sec * fs)
        generated_tone[:fading_samples] *= np.linspace(0, 1, fading_samples)
        generated_tone[-fading_samples:] *= np.linspace(1, 0, fading_samples)

    generated_tone = gain * normalize(generated_tone)
    return generated_tone


def generate_fm_synthesized_tone(pitch=69, modulation_frequency=0, modulation_index=0, amp=1, dur=1, fs=44100,
                                 f_tuning=440, fade_dur=0.01):
    # TODO: adjust to generate_tone_additive_synthesis
    """Generate fm synthesized tone

    Args:
        pitch: pitch of the synthesized tone
        modulation_frequency: frequency to modulate
        modulation_index: strength of modulation
        amp: amplitude of resulting signal
        dur: duration (in seconds)
        Fs: Sampling rate
        f_tuning: Tuning frequency
        fade_dur: Duration of fade in and fade out (to avoid clicks)

    Returns:
        y: synthesized tone
        t: time axis (in seconds)
    """
    N = int(dur * fs)
    t = np.arange(N) / fs
    freq = f_tuning * 2 ** ((pitch - 69) / 12)
    y = np.sin(2 * np.pi * freq * t + modulation_index * np.sin(2 * np.pi * modulation_frequency * t))
    if not fade_dur == 0:
        fade_samples = int(fade_dur * fs)
        y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amp * (y / np.max(y))
    return y


def generate_sinusoid(frequency: float = 440.0,
                      amp: float = 1.0,
                      dur: float = 1.0,
                      fs: int = 22050,
                      fade_dur: float = 0.01) -> np.ndarray:
    """
    This function generates a sinusoid.

    Parameters
    ----------
    frequency : float, default = 440.0 Hz
        Frequency in Hertz for sinusoid
    amp : float, default = 1.0
        Amplitude for sinusoid
    dur : float, default = 1.0 s
        Duration in seconds for sinusoid
    fade_dur : float, default = 0.01 s
        Duration in seconds of fade in and fade out
    fs : int, default = 22050
        Sampling rate
    Returns
    -------
    y : np.ndarray
        sinusoid
    """
    N = int(dur * fs)
    t = np.arange(N) / fs

    y = np.sin(2 * np.pi * frequency * t)
    if not fade_dur == 0:
        fade_samples = int(fade_dur * fs)
        y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amp * (y / np.max(y))
    return y


def envelop_signal(signal: np.ndarray, attack_time: float = 0, decay_time: float = 0, sustain_level: float = 0,
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
