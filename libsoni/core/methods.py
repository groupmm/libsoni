import numpy as np
from typing import Tuple

from libsoni.util.utils import fade_signal, smooth_weights, normalize_signal, pitch_to_frequency


def generate_click(pitch: int = 69,
                   amplitude: float = 1.0,
                   tuning_frequency: float = 440.0,
                   click_fading_duration: float = 0.2,
                   fs: int = 22050) -> np.ndarray:
    """Generates a click signal.

    Parameters
    ----------
    pitch : int, default = 69
        Pitch for colored click.

    amplitude : float, default = 1.0
        Amplitude of click signal.

    click_fading_duration : float, default = 0.2
        Fading duration of click signal, in seconds.

    tuning_frequency : float, default = 440.0
        Tuning frequency, in Hertz.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    click : np.ndarray
        Generated click signal.
    """

    assert 0 <= pitch <= 127, f'Pitch is out of range [0,127].'

    click_frequency = pitch_to_frequency(pitch=pitch, tuning_frequency=tuning_frequency)
    angular_frequency = 2 * np.pi * click_frequency / fs
    click = np.logspace(0, -10, num=int(fs * click_fading_duration), base=2.0)
    click *= amplitude * np.sin(angular_frequency * np.arange(len(click)))

    return click


def generate_sinusoid(frequency: float = 440.0,
                      phase: float = 0.0,
                      amplitude: float = 1.0,
                      duration: float = 1.0,
                      fading_duration: float = 0.01,
                      fs: int = 22050) -> np.ndarray:
    """Generates sinusoid.

    Parameters
    ----------
    frequency: float, default: 440.0
        Frequency of sinusoid, in Hertz.

    phase: float, default: 0.0
        Phase of sinusoid.

    amplitude: float, default: 1.0
        Amplitude of sinusoid.

    duration: float, default: 1.0
        Duration of generated signal, in seconds.

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    sinusoid: np.ndarray
        Generated sinusoid.
    """
    sinusoid = amplitude * np.sin((2 * np.pi * frequency * (np.arange(int(duration * fs)) / fs)) + phase)
    sinusoid = fade_signal(signal=sinusoid, fs=fs, fading_duration=fading_duration)

    return sinusoid


def generate_shepard_tone(pitch_class: int = 0,
                          pitch_range: Tuple[int, int] = (20, 108),
                          filter: bool = False,
                          f_center: float = 440.0,
                          octave_cutoff: int = 1,
                          gain: float = 1.0,
                          duration: float = 1.0,
                          tuning_frequency: float = 440,
                          fading_duration: float = 0.05,
                          fs: int = 22050) -> np.ndarray:
    """Generates shepard tone.

    The sound can be changed either by the filter option or by the specified pitch-range.
    Both options can also be used in combination.
    Using the filter option shapes the spectrum like a bell curve centered around the center frequency,
    while the octave cutoff determines at which octave the amplitude of the corresponding sinusoid is 0.5.

    Parameters
    ----------
    pitch_class: int, default: 0
        Pitch class for shepard tone.

    pitch_range: Tuple[int, int], default = [20,108]
        Determines the pitch range to encounter for shepard tones.

    filter: bool, default: False
        Enables filtering of shepard tones.

    f_center : float, default: 440.0
        Determines filter center frequency, in Hertz.

    octave_cutoff: int, default: 1
        Determines the width of the filter.

    gain: float, default: 1.0
        Gain of shepard tone.

    duration: float, default: 1.0
        Determines duration of shepard tone, in seconds.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    shepard_tone: np.ndarray
        Generated shepard tone.
    """

    assert 0 <= pitch_class <= 11, f'Pitch class is out of range [0,11].'
    assert all(0 <= p <= 127 for p in pitch_range), f'Pitch range has to be defined within [0,127].'

    pitches = pitch_class + np.arange(11) * 12
    mask = (pitches >= pitch_range[0]) & (pitches <= pitch_range[1])
    shepard_frequencies = tuning_frequency * 2 ** ((pitches[mask] - 69) / 12)
    shepard_tone = np.zeros(int(duration * fs))

    if filter:
        f_log = 2 * np.logspace(1, 4, 20000)
        f_lin = np.linspace(20, 20000, 20000)
        f_center_lin = np.argmin(np.abs(f_log - f_center))
        weights = np.exp(- (f_lin - f_center_lin) ** 2 / (1.4427 * ((octave_cutoff * 2) * 1000) ** 2))

        for shepard_frequency in shepard_frequencies:
            shepard_tone += weights[np.argmin(np.abs(f_log - shepard_frequency))] * \
                            np.sin(2 * np.pi * shepard_frequency * np.arange(int(duration * fs)) / fs)

    else:
        for shepard_frequency in shepard_frequencies:
            shepard_tone += np.sin(2 * np.pi * shepard_frequency * np.arange(int(duration * fs)) / fs)

    shepard_tone = fade_signal(signal=shepard_tone, fs=fs, fading_duration=fading_duration)
    shepard_tone = normalize_signal(shepard_tone) * gain

    return shepard_tone


def generate_tone_additive_synthesis(pitch: int = 69,
                                     partials: np.ndarray = np.array([1]),
                                     partials_amplitudes: np.ndarray = None,
                                     partials_phase_offsets: np.ndarray = None,
                                     gain: float = 1.0,
                                     duration: float = 1.0,
                                     tuning_frequency: float = 440,
                                     fading_duration: float = 0.05,
                                     fs: int = 22050) -> np.ndarray:
    """Generates tone signal using additive synthesis.

    The sound can be customized using parameters partials, partials_amplitudes and partials_phase_offsets.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the generated tone.

    partials: np.ndarray, default = [1]
        Array containing the desired partials of the fundamental frequencies for sonification.
        An array [1] leads to sonification with only the fundamental frequency,
        while an array [1,2] leads to sonification with the fundamental frequency and twice the fundamental frequency.

    partials_amplitudes: np.ndarray, default = None
        Array containing the amplitudes for partials.
        An array [1,0.5] causes the first partial to have amplitude 1,
        while the second partial has amplitude 0.5.
        When not defined, the amplitudes for all partials are set to 1.

    partials_phase_offsets: np.ndarray, default = None
        Array containing the phase offsets for partials.
        When not defined, the phase offsets for all partials are set to 0.

    gain: float, default = 1.0
        Gain of generated tone.

    duration: float, default: 1.0
        Determines duration of shepard tone, given in seconds.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, given in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    generated_tone: np.ndarray
        Generated tone signal.
    """

    assert 0 <= pitch <= 127, f'Pitch is out of range [0,127].'

    partials_amplitudes = np.ones(len(partials)) if partials_amplitudes is None else partials_amplitudes
    partials_phase_offsets = np.zeros(len(partials)) if partials_phase_offsets is None else partials_phase_offsets

    assert len(partials) == len(partials_amplitudes) == len(partials_phase_offsets), \
        f'Arrays partials, partials_amplitudes and partials_phase_offsets must be of equal length.'

    generated_tone = np.zeros(int(duration * fs))
    pitch_frequency = pitch_to_frequency(pitch=pitch, tuning_frequency=tuning_frequency)

    for partial, partial_amplitude, partials_phase_offset in zip(partials, partials_amplitudes, partials_phase_offsets):
        generated_tone += partial_amplitude * np.sin(2 * np.pi * pitch_frequency * partial * (np.arange(int(duration * fs)) / fs) + partials_phase_offset)

    generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_duration=fading_duration)

    return generated_tone * gain


def generate_tone_fm_synthesis(pitch: int = 69,
                               modulation_rate_relative: float = 0.0,
                               modulation_amplitude: float = 0.0,
                               gain: float = 1.0,
                               duration: float = 1.0,
                               tuning_frequency: float = 440.0,
                               fading_duration: float = 0.05,
                               fs: int = 22050) -> np.ndarray:
    """Generates tone signal using frequency modulation synthesis.

    The sound can be customized using parameters modulation_rate_relative and modulation_amplitude.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the synthesized tone.

    modulation_rate_relative: float, default = 0.0
        Determines the modulation frequency as multiple or fraction of the frequency for the given pitch.

    modulation_amplitude: float, default = 0.0
        Determines the amount of modulation in the generated signal.

    gain: float, default = 1.0
        Gain for generated signal

    duration: float, default: 1.0
        Determines duration of shepard tone, given in seconds.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, given in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    generated_tone: np.ndarray
        Generated tone signal.
    """

    assert 0 <= pitch <= 127, f'Pitch is out of range [0,127].'

    pitch_frequency = pitch_to_frequency(pitch=pitch, tuning_frequency=tuning_frequency)
    generated_tone = np.sin(2 * np.pi * pitch_frequency * (np.arange(int(duration * fs))) / fs +
                            modulation_amplitude * np.sin(2 * np.pi * pitch_frequency * modulation_rate_relative *
                                                          (np.arange(int(duration * fs)))))
    generated_tone = gain * fade_signal(signal=generated_tone, fs=fs, fading_duration=fading_duration)

    return generated_tone


def generate_tone_wavetable(pitch: int = 69,
                            wavetable: np.ndarray = None,
                            gain: float = 1.0,
                            duration: float = 1.0,
                            tuning_frequency: float = 440.0,
                            fading_duration: float = 0.05,
                            fs: int = 22050) -> np.ndarray:
    """Generates tone using wavetable synthesis.

    The sound depends on the given wavetable.

    Parameters
    ----------
    pitch: int, default = 69
        Pitch of the synthesized tone.

    wavetable: np.ndarray, default = None
        Wavetable to be resampled.

    gain: float, default = 1.0
        Gain for generated signal

    duration: float, default: 1.0
        Determines duration of tone, given in seconds.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    generated_tone: np.ndarray
        Generated signal
    """

    assert 0 <= pitch <= 127, f'Pitch is out of range [0,127].'

    generated_tone = []
    pitch_frequency = pitch_to_frequency(pitch=pitch, tuning_frequency=tuning_frequency)
    current_sample = 0

    while len(generated_tone) < int(duration * fs):
        current_sample += int(pitch_frequency)
        current_sample = current_sample % wavetable.size
        generated_tone.append(wavetable[current_sample])
        current_sample += 1

    generated_tone = np.array(generated_tone)
    generated_tone = gain * fade_signal(signal=generated_tone, fs=fs, fading_duration=fading_duration)

    return generated_tone


def generate_tone_instantaneous_phase(frequency_vector: np.ndarray,
                                      gain_vector: np.ndarray = None,
                                      partials: np.ndarray = np.array([1]),
                                      partials_amplitudes: np.ndarray = None,
                                      partials_phase_offsets: np.ndarray = None,
                                      fading_duration: float = 0.05,
                                      fs: int = 22050) -> np.ndarray:
    """Generates signal out of instantaneous frequency.

    The sound can be customized using parameters partials, partials_amplitudes and partials_phase_offsets.

    Parameters
    ----------
    frequency_vector: np.ndarray
        Array containing sample-wise instantaneous frequencies.

    gain_vector: np.ndarray, default = None
        Array containing sample-wise gains.

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

    fading_duration: float, default: 0.01
        Determines duration of fade-in and fade-out, given in seconds.

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    generated_tone: np.ndarray
        Generated signal
    """
    partials_amplitudes = np.ones(len(partials)) if partials_amplitudes is None else partials_amplitudes
    partials_phase_offsets = np.zeros(len(partials)) if partials_phase_offsets is None else partials_phase_offsets

    assert len(partials) == len(partials_amplitudes) == len(partials_phase_offsets), \
        'Partials, Partials_amplitudes and Partials_phase_offsets must be of equal length.'

    generated_tone = np.zeros_like(frequency_vector)

    if gain_vector is None:
        gain_vector = np.ones_like(frequency_vector)

    else:
        gain_vector = smooth_weights(weights=gain_vector, fading_samples=60)

    phase = 0
    phase_result = []

    for frequency, gain in zip(frequency_vector, gain_vector):
        phase_step = 2 * np.pi * frequency / fs
        phase += phase_step
        phase_result.append(phase)

    phase_result = np.asarray(phase_result)

    for partial, partial_amplitude, partials_phase_offset in zip(partials, partials_amplitudes, partials_phase_offsets):
        generated_tone += np.sin((phase_result + partials_phase_offset) * partial) * partial_amplitude

    generated_tone = generated_tone * gain_vector
    generated_tone = fade_signal(signal=generated_tone, fs=fs, fading_duration=fading_duration)

    return generated_tone
