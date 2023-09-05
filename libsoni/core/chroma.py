import numpy as np
from typing import Tuple

from libsoni.util.utils import normalize_signal, fade_signal
from libsoni.core.methods import generate_shepard_tone


def sonify_chromagram(chromagram: np.ndarray,
                      H: int = 0,
                      pitch_range: Tuple[int, int] = (20, 108),
                      filter: bool = False,
                      f_center: float = 440.0,
                      octave_cutoff: int = 1,
                      sonification_duration: int = None,
                      fade_duration: float = 0.05,
                      normalize: bool = True,
                      fs: int = 22050,
                      tuning_frequency: float = 440.0) -> np.ndarray:
    """Sonify chromagram

        Parameters
        ----------
        chromagram: np.ndarray
            Chromagram to sonify.
        H: int, default = 0
            Hop size of STFT, used to calculate chromagram.
        pitch_range: Tuple[int, int], default = [20,108]
            pitches to encounter in shepard tone
        filter: bool, default: False
            decides, if shepard tones are filtered or not
        f_center : float, default: 440.0
            center_frequency in Hertz for bell-shaped filter
        octave_cutoff: int, default: 1
            determines, at which multiple of f_center, the harmonics get attenuated by 2.
        sonification_duration: int, default = None
            Duration of audio, given in samples
        fade_duration: float, default = 0.05
            Duration of fade-in and fade-out at beginning and end of the sonification, given in seconds.
        fs: int, default: 44100
            sampling rate in Samples/second
        normalize: bool, default = True
            Decides, if output signal is normalized to [-1,1].
        fs: int, default = 22050
            Sampling rate, in samples per seconds.
        tuning_frequency : float, default = 440
            Tuning frequency.

        Returns
        -------
            y: synthesized tone
        """
    assert chromagram.shape[0] == 12, f'The chromagram must have shape 12xN.'

    # Compute frame rate
    frame_rate = fs / H

    # Determine length of sonification
    num_samples = sonification_duration if sonification_duration is not None else int(chromagram.shape[1] * fs / frame_rate)

    # Compute length of fading in samples
    fade_values = int(H / 8)

    # Initialize sonification
    chroma_sonification = np.zeros(num_samples)

    for pitch_class in range(12):
        if np.sum(np.abs(chromagram[pitch_class, :])) > 0:
            weighting_vector = np.repeat(chromagram[pitch_class, :], 512)
            weighting_vector_smoothed = np.copy(weighting_vector)
            for i in range(1, len(weighting_vector)):
                if weighting_vector[i] != weighting_vector[i - 1]:
                    frequency = 1
                    amplitude = (np.abs(weighting_vector[i - 1] - weighting_vector[i])) / 2

                    x = np.linspace(-1 * (np.pi / 2), np.pi / 2, fade_values) * -1 * np.sign(weighting_vector[i - 1] - weighting_vector[i])

                    y = amplitude * np.sin(frequency * x) + (weighting_vector[i - 1] + weighting_vector[i]) / 2

                    weighting_vector_smoothed[i - int(fade_values / 2):i - int(fade_values / 2) + len(y)] = y

            shepard_tone = generate_shepard_tone(pitch_class=pitch_class,
                                                 pitch_range=pitch_range,
                                                 filter=filter,
                                                 f_center=f_center,
                                                 octave_cutoff=octave_cutoff,
                                                 gain=1,
                                                 duration_sec=num_samples / fs,
                                                 fs=fs,
                                                 f_tuning=tuning_frequency,
                                                 fading_sec=0)

            chroma_sonification += (shepard_tone * weighting_vector_smoothed)

    chroma_sonification = fade_signal(chroma_sonification, fs=fs, fading_sec=fade_duration)

    chroma_sonification = normalize_signal(chroma_sonification) if normalize else chroma_sonification

    return chroma_sonification
