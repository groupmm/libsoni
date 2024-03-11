import numpy as np
from typing import Tuple

from libsoni.util.utils import normalize_signal, fade_signal, smooth_weights
from libsoni.core.methods import generate_shepard_tone


def sonify_chroma_vector(chroma_vector: np.ndarray,
                         pitch_range: Tuple[int, int] = (20, 108),
                         filter: bool = False,
                         f_center: float = 440.0,
                         octave_cutoff: int = 1,
                         tuning_frequency: float = 440.0,
                         fading_duration: float = 0.05,
                         sonification_duration: int = None,
                         normalize: bool = True,
                         fs: int = 22050) -> np.ndarray:
    """Sonifies a chroma vector using sound synthesis based on shepard tones.

    The sound can be changed either by the filter option or by the specified pitch-range.
    Both options can also be used in combination. Using the filter option shapes the spectrum
    like a bell curve centered around the center frequency, while the octave cutoff determines
    at which octave the amplitude of the corresponding sinusoid is 0.5.

    Parameters
    ----------
    chroma_vector: np.ndarray
        Chroma vector to sonify.

    pitch_range: Tuple[int, int], default = [20,108]
        Determines the pitches to encounter for shepard tones.

    filter: bool, default: False
        Enables filtering of shepard tones.

    f_center : float, default: 440.0
        Determines filter center frequency, in Hertz.

    octave_cutoff: int, default: 1
        Determines the width of the filter.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    chroma_sonification: np.ndarray
        Sonified chroma vector.
    """

    assert len(chroma_vector) == 12, f'The chroma vector must have length 12.'

    # Determine length of sonification
    num_samples = sonification_duration

    # Initialize sonification
    chroma_sonification = np.zeros(num_samples)

    for pitch_class in range(12):
        if chroma_vector[pitch_class] > 0:
            shepard_tone = generate_shepard_tone(pitch_class=pitch_class,
                                                 pitch_range=pitch_range,
                                                 filter=filter,
                                                 f_center=f_center,
                                                 octave_cutoff=octave_cutoff,
                                                 gain=chroma_vector[pitch_class],
                                                 duration=num_samples / fs,
                                                 tuning_frequency=tuning_frequency,
                                                 fading_duration=fading_duration,
                                                 fs=fs)
            chroma_sonification += shepard_tone

    chroma_sonification = fade_signal(chroma_sonification, fading_duration=fading_duration, fs=fs)
    chroma_sonification = normalize_signal(chroma_sonification) if normalize else chroma_sonification

    return chroma_sonification


def sonify_chromagram(chromagram: np.ndarray,
                      H: int = 0,
                      pitch_range: Tuple[int, int] = (20, 108),
                      filter: bool = False,
                      f_center: float = 440.0,
                      octave_cutoff: int = 1,
                      tuning_frequency: float = 440.0,
                      fading_duration: float = 0.05,
                      sonification_duration: int = None,
                      normalize: bool = True,
                      fs: int = 22050) -> np.ndarray:
    """Sonifies a chromagram using sound synthesis based on shepard tones.

    The sound can be changed either by the filter option or by the specified pitch-range.
    Both options can also be used in combination.
    Using the filter option shapes the spectrum like a bell curve centered around the center frequency,
    while the octave cutoff determines at which octave the amplitude of the corresponding sinusoid is 0.5.

    Parameters
    ----------
    chromagram: np.ndarray
        Chromagram to sonify.

    H: int, default = 0
        Hop size of STFT used to calculate chromagram.

    pitch_range: Tuple[int, int], default = [20,108]
        Determines the pitch range to encounter for shepard tones.

    filter: bool, default: False
        Enables filtering of shepard tones.

    f_center : float, default: 440.0
        Determines filter center frequency, in Hertz.

    octave_cutoff: int, default: 1
        Determines the width of the filter.
        For octave_cutoff of 1, the magnitude of the filter reaches 0.5 at half the center_frequency and twice the center_frequency.

    tuning_frequency: float, default: 440.0
        Tuning frequency, in Hertz.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    chroma_sonification: np.ndarray
        Sonified chromagram.
    """

    assert chromagram.shape[0] == 12, f'The chromagram must have shape 12xN.'

    # Compute frame rate
    frame_rate = fs / H

    # Determine length of sonification
    num_samples = sonification_duration if sonification_duration is not None else int(
        chromagram.shape[1] * fs / frame_rate)

    # Initialize sonification
    chroma_sonification = np.zeros(num_samples)

    for pitch_class in range(12):
        if np.sum(np.abs(chromagram[pitch_class, :])) > 0:
            weighting_vector = np.repeat(chromagram[pitch_class, :], H)
            weighting_vector_smoothed = smooth_weights(weights=weighting_vector, fading_samples=int(H / 8))
            shepard_tone = generate_shepard_tone(pitch_class=pitch_class,
                                                 pitch_range=pitch_range,
                                                 filter=filter,
                                                 f_center=f_center,
                                                 octave_cutoff=octave_cutoff,
                                                 gain=1,
                                                 duration=num_samples / fs,
                                                 tuning_frequency=tuning_frequency,
                                                 fading_duration=fading_duration,
                                                 fs=fs)
            chroma_sonification += (shepard_tone * weighting_vector_smoothed)

    chroma_sonification = fade_signal(chroma_sonification, fading_duration=fading_duration, fs=fs)
    chroma_sonification = normalize_signal(chroma_sonification) if normalize else chroma_sonification

    return chroma_sonification
