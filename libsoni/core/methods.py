import numpy as np
from typing import Tuple

from libsoni.util.utils import fade_signal


def generate_click(pitch: int = 69,
                   amplitude: float = 1.0,
                   fading_duration: float = 0.2,
                   fs: int = 22050,
                   tuning_frequency: float = 440.0) -> np.ndarray:
    """Returns a click signal.

    Parameters
    ----------
    pitch : int, default = 69
        Pitch for colored click.
    amplitude : float, default = 1.0
        Amplitude of click signal.
    fading_duration : float, default = 0.2
        Fading duration of click signal.
    fs : int, default = 22050
        Sampling rate.
    tuning_frequency : float, default = 440
        Tuning frequency.
    Returns
    -------
    click : np.ndarray
        Click signal.
    """
    click_freq = tuning_frequency * 2 ** ((pitch - 69) / 12)
    angular_freq = 2 * np.pi * click_freq / float(fs)
    click = np.logspace(0, -10, num=int(fs * fading_duration), base=2.0)
    click *= np.sin(angular_freq * np.arange(len(click)))
    click *= amplitude
    return click


def generate_shepard_tone(pitch_class: int = 0,
                          pitch_range: Tuple[np.ndarray, int] = (0, 127),
                          filter: bool = False,
                          f_center: float = 440.0,
                          octave_cutoff: int = 1,
                          gain: float = 1.0,
                          duration_sec: float = 1.0,
                          fs: int = 22050,
                          f_tuning: float = 440,
                          fading_sec: float = 0.01,
                          ) -> np.ndarray:
    """Generate shepard tone

    Parameters
    ----------
    pitch_class: int, default: 0
        pitch class of the synthesized tone
    pitch_range: Tuple[int, int], default = [20,108]
        pitches to encounter in shepard tone
    filter: bool, default: False
        decides, if shepard tones are filtered or not
    f_center : float, default: 440.0
        center_frequency in Hertz for bell-shaped filter
    octave_cutoff: int, default: 1
        determines, at which multiple of f_center, the harmonics get attenuated by 2.
    gain: float, default: 1.0
        gain of resulting signal
    duration_sec: float, default: 1.0
        duration of generated signal (in seconds)
    fs: int, default: 44100
        sampling rate in Samples/second
    f_tuning: float, default: 440.0
        tuning frequency (in Hz)
    fading_sec: float, default: 0.01
        sonification_duration (in seconds) of fade in and fade out (to avoid clicks)

    Returns
    -------
        y: synthesized tone
    """
    assert 0 <= pitch_class <= 11, "pitch class out of range"

    num_samples = int(duration_sec * fs)

    t = np.arange(num_samples) / fs

    pitches = pitch_class + np.arange(11) * 12

    mask = (pitches >= pitch_range[0]) & (pitches <= pitch_range[1])

    freqs = f_tuning * 2 ** ((pitches[mask] - 69) / 12)

    generated_tone = np.zeros(num_samples)

    if filter:
        f_log = 2 * np.logspace(1, 4, 20000)
        f_lin = np.linspace(20, 20000, 20000)
        f_center_lin = np.argmin(np.abs(f_log - f_center))
        weights = np.exp(- (f_lin - f_center_lin) ** 2 / (1.4427 * ((octave_cutoff * 2) * 1000) ** 2))

        for freq in freqs:
            generated_tone += weights[np.argmin(np.abs(f_log - freq))] * np.sin(2 * np.pi * freq * t)

    else:
        for freq in freqs:
            generated_tone += np.sin(2 * np.pi * freq * t)

    if not fading_sec == 0:
        generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_sec=fading_sec)

    return generated_tone * gain


def generate_tone_additive_synthesis(pitch: int = 69,
                                     partials: np.ndarray = np.array([1]),
                                     partials_amplitudes: np.ndarray = None,
                                     partials_phase_offsets: np.ndarray = None,
                                     gain: float = 1.0,
                                     duration_sec: float = 1.0,
                                     fs: int = 22050,
                                     f_tuning: float = 440,
                                     fading_sec: float = 0.01) -> np.ndarray:
    """Generates signal using additive synthesis.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the synthesized tone.
    partials: np.ndarray, default = [1]
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

    assert len(partials) == len(partials_amplitudes) == len(partials_phase_offsets), \
        'Partials, Partials_amplitudes and Partials_phase_offsets must be of equal length.'

    num_samples = int(duration_sec * fs)

    t = np.arange(num_samples) / fs

    generated_tone = np.zeros(len(t))

    pitch_frequency = f_tuning * 2 ** ((pitch - 69) / 12)

    if num_samples < 2 * int(fading_sec * fs):
        fading_sec = 0

    for partial, partial_amplitude, partials_phase_offset in zip(partials, partials_amplitudes, partials_phase_offsets):
        generated_tone += partial_amplitude * np.sin(2 * np.pi * pitch_frequency * partial * t + partials_phase_offset)

    if not fading_sec == 0:
        generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_sec=fading_sec)

    return generated_tone * gain


def generate_tone_fm_synthesis(pitch: int = 69,
                               mod_rate_relative: float = 0.0,
                               mod_amp: float = 0.0,
                               gain: float = 1.0,
                               duration_sec: float = 1.0,
                               fs: int = 22050,
                               f_tuning: float = 440,
                               fading_sec: float = 0.01) -> np.ndarray:
    """Generates signal using frequency modulation synthesis.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the synthesized tone.
    mod_rate_relative: float, default = 0.0
        Determines the modulation frequency as multiple or fraction of the frequency for the given pitch.
    mod_amp: float, default = 0.0
        Determines the amount of modulation in the generated signal.
    gain: float, default = 1.0
        Gain for generated signal
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
    N = int(duration_sec * fs)
    t = np.arange(N) / fs
    freq = f_tuning * 2 ** ((pitch - 69) / 12)
    generated_tone = np.sin(2 * np.pi * freq * t + mod_amp * np.sin(2 * np.pi * freq * mod_rate_relative * t))
    if not fading_sec == 0:
        generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_sec=fading_sec)

    return generated_tone * gain


def generate_tone_wavetable(pitch: int = 69,
                            wavetable: np.ndarray = None,
                            gain: float = 1.0,
                            duration_sec: float = 1.0,
                            fs: int = 22050,
                            f_tuning: float = 440,
                            fading_sec: float = 0.01) -> np.ndarray:
    """Generate tone by resampling a wavetable
    pitch: int, default = 69
        Pitch of the synthesized tone.
    modulation_frequency_factor: float, default = 0.0
        Determines the modulation frequency as multiple or fraction of the frequency for the given pitch.
    modulation_index: float, default = 0.0
        Determines the amount of modulation in the generated signal.
    gain: float, default = 1.0
        Gain for generated signal
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
    num_samples = int(duration_sec * fs)
    generated_tone = []
    freq = f_tuning * 2 ** ((pitch - 69) / 12)
    current_sample = 0
    while len(generated_tone) < num_samples:
        current_sample += int(freq)
        current_sample = current_sample % wavetable.size
        generated_tone.append(wavetable[current_sample])
        current_sample += 1
    generated_tone = np.array(generated_tone)

    if not fading_sec == 0:
        generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_sec=fading_sec)

    return generated_tone * gain

def generate_tone_instantaneous_phase(frequency_vector: np.ndarray,
                                      partials: np.ndarray = np.array([1]),
                                      partials_amplitudes: np.ndarray = None,
                                      partials_phase_offsets: np.ndarray = None,
                                      gain_vector: np.ndarray = None,
                                      duration_sec: float = 1.0,
                                      fs: int = 22050,
                                      f_tuning: float = 440,
                                      fading_sec: float = 0.01) -> np.ndarray:

    generated_tone = np.zeros(len(gain_vector))

    phase = 0
    phase_result = []
    for frequency, gain in zip(frequency_vector, gain_vector):
        phase_step = 2 * np.pi * frequency * 1 / fs
        phase += phase_step
        phase_result.append(phase)

    generated_tone += np.sin(phase_result)
    #print(generated_tone)
    #for partial, partial_amplitude in zip(partials, partials_amplitudes):

        #generated_tone += np.sin(phase_result*partial) * partial_amplitude

    generated_tone = np.multiply(generated_tone, gain_vector)

    return generated_tone
