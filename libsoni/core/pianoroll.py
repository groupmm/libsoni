import numpy as np
import pandas as pd

from libsoni.util.utils import format_df, warp_sample, normalize_signal, fade_signal
from libsoni.core.methods import generate_click, generate_tone_additive_synthesis, generate_tone_fm_synthesis


def sonify_pianoroll_additive_synthesis(pianoroll_df: pd.DataFrame,
                                        partials: np.ndarray = np.array([1]),
                                        partials_amplitudes: np.ndarray = None,
                                        partials_phase_offsets: np.ndarray = None,
                                        tuning_frequency: float = 440.0,
                                        signal_fading_duration: float = 0.05,
                                        note_fading_duration: float = 0.01,
                                        sonification_duration: float = None,
                                        normalize: bool = True,
                                        fs: int = 22050) -> np.ndarray:
    """Sonifies a piano-roll with additive synthesis.

    The DataFrame representation is assumed to contain row-wise pitch events described by start, duration or end
    and the corresponding pitch.
    The sonification is based on additive synthesis, where parameters partials, partials_amplitudes and
    partials_phase_offsets can be used to shape the sound.

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe containing pitch-event information.

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

    tuning_frequency: float, default = 440.0
        Tuning frequency, in Hertz.

    sonification_duration: float, default = None
        Determines duration of sonification, in seconds.

    signal_fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the final sonification, in seconds.

    note_fading_duration: float, default = 0.01
        Determines duration of fade-in and fade-out at beginning and end of each note event, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified piano-roll.
    """

    pianoroll_df, pianoroll_sonification = __init_pianoroll_sonification(pianoroll_df, fs, sonification_duration)

    if 'velocity' in list(pianoroll_df.columns) and pianoroll_df['velocity'].max() > 1:
        pianoroll_df['velocity'] /= pianoroll_df['velocity'].max()

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        amplitude = r['velocity'] if 'velocity' in r else 1.0
        pianoroll_sonification[start_samples:start_samples + duration_samples] += \
            generate_tone_additive_synthesis(pitch=r['pitch'],
                                             partials=partials,
                                             partials_amplitudes=partials_amplitudes,
                                             partials_phase_offsets=partials_phase_offsets,
                                             gain=amplitude,
                                             duration=r['duration'],
                                             tuning_frequency=tuning_frequency,
                                             fading_duration=note_fading_duration,
                                             fs=fs)

    pianoroll_sonification = fade_signal(pianoroll_sonification, fs=fs, fading_duration=signal_fading_duration)
    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def sonify_pianoroll_clicks(pianoroll_df: pd.DataFrame,
                            tuning_frequency: float = 440.0,
                            sonification_duration: float = None,
                            signal_fading_duration: float = 0.05,
                            normalize: bool = True,
                            fs: int = 22050) -> np.ndarray:
    """Sonifies a piano-roll with clicks.

    The DataFrame representation is assumed to contain row-wise pitch events described by start, duration or end
    and the corresponding pitch.
    For sonification, coloured clicks are used.

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe containing pitch-event information.

    tuning_frequency: float, default = 440.0
        Tuning Frequency, in Hertz

    sonification_duration: float, default = None
        Determines duration of sonification, in seconds.

    signal_fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the final sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """

    pianoroll_df, pianoroll_sonification = __init_pianoroll_sonification(pianoroll_df, fs, sonification_duration)

    if 'velocity' in list(pianoroll_df.columns) and pianoroll_df['velocity'].max() > 1:
        pianoroll_df['velocity'] /= pianoroll_df['velocity'].max()

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        amplitude = r['velocity'] if 'velocity' in r else 1.0
        pianoroll_sonification[start_samples:start_samples + duration_samples] += \
            generate_click(pitch=r['pitch'],
                           amplitude=amplitude,
                           tuning_frequency=tuning_frequency,
                           click_fading_duration=r['duration'],
                           fs=fs)

    pianoroll_sonification = fade_signal(pianoroll_sonification, fs=fs, fading_duration=signal_fading_duration)
    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def sonify_pianoroll_sample(pianoroll_df: pd.DataFrame,
                            sample: np.ndarray = None,
                            reference_pitch: int = 69,
                            sonification_duration: float = None,
                            signal_fading_duration: float = 0.05,
                            note_fading_duration: float = 0.01,
                            normalize: bool = True,
                            fs: int = 22050) -> np.ndarray:
    """Sonifies a piano-roll based on custom audio samples.

    The DataFrame representation is assumed to contain row-wise pitch events described by start, duration or end
    and the corresponding pitch. For sonification, warped versions of the given sample are used.

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe containing pitch-event information.

    sample: np.ndarray
        Sample to use for sonification.

    reference_pitch: int, default = 69
        Original pitch of the sample.

    sonification_duration: float, default = None
        Determines duration of sonification, in seconds.

    signal_fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the final sonification, in seconds.

    note_fading_duration: float, default = 0.01
        Determines duration of fade-in and fade-out at beginning and end of each note event, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified piano-roll.
    """

    pianoroll_df, pianoroll_sonification = __init_pianoroll_sonification(pianoroll_df, fs, sonification_duration)

    if 'velocity' in list(pianoroll_df.columns) and pianoroll_df['velocity'].max() > 1:
        pianoroll_df['velocity'] /= pianoroll_df['velocity'].max()

    else:
        pianoroll_df['velocity'] = 1

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        warped_sample = warp_sample(sample=sample,
                                    reference_pitch=reference_pitch,
                                    target_pitch=r['pitch'],
                                    target_duration_sec=r['duration'],
                                    gain=r['velocity'],
                                    fading_duration=note_fading_duration,
                                    fs=fs)
        pianoroll_sonification[start_samples:start_samples + len(warped_sample)] += warped_sample

    pianoroll_sonification = fade_signal(pianoroll_sonification, fs=fs, fading_duration=signal_fading_duration)
    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def sonify_pianoroll_fm_synthesis(pianoroll_df: pd.DataFrame,
                                  mod_rate_relative: float = 0.0,
                                  mod_amp: float = 0.0,
                                  tuning_frequency: float = 440.0,
                                  sonification_duration: float = None,
                                  signal_fading_duration: float = 0.05,
                                  note_fading_duration: float = 0.01,
                                  normalize: bool = True,
                                  fs: int = 22050) -> np.ndarray:
    """Sonifies a piano-roll with frequency modulation (FM) synthesis.

    The DataFrame representation is assumed to contain row-wise pitch events described by start, duration or end
    and the corresponding pitch. The sonification is based on FM synthesis, where parameters mod_rate_relative and
    mod_amp can be used to shape the sound.

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe containing pitch-event information.

    mod_rate_relative: float, default = 0.0
        Determines the modulation frequency as multiple or fraction of the frequency for the given pitch.

    mod_amp: float, default = 0.0
        Determines the amount of modulation in the generated signal.

    tuning_frequency: float, default = 440.0
        Tuning frequency in Hertz.

    sonification_duration: float, default = None
        Determines duration of sonification, in seconds.

    signal_fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the final sonification, in seconds.

    note_fading_duration: float, default = 0.01
        Determines duration of fade-in and fade-out at beginning and end of each note event, in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified piano-roll.
    """
    pianoroll_df, pianoroll_sonification = __init_pianoroll_sonification(pianoroll_df, fs, sonification_duration)

    if 'velocity' in list(pianoroll_df.columns) and pianoroll_df['velocity'].max() > 1:
        pianoroll_df['velocity'] /= pianoroll_df['velocity'].max()

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        amplitude = r['velocity'] if 'velocity' in r else 1.0
        pianoroll_sonification[start_samples:start_samples + duration_samples] += \
            generate_tone_fm_synthesis(pitch=r['pitch'],
                                       modulation_rate_relative=mod_rate_relative,
                                       modulation_amplitude=mod_amp,
                                       gain=amplitude,
                                       duration=r['duration'],
                                       tuning_frequency=tuning_frequency,
                                       fading_duration=note_fading_duration,
                                       fs=fs)

    pianoroll_sonification = fade_signal(pianoroll_sonification, fs=fs, fading_duration=signal_fading_duration)
    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def __init_pianoroll_sonification(pianoroll_df: pd.DataFrame,
                                  fs: int,
                                  sonification_duration: float = None):

    pianoroll_df = format_df(pianoroll_df)
    num_samples = int(pianoroll_df['end'].max() * fs)

    if sonification_duration is None:
        sonification_duration = pianoroll_df['end'].max()
    sonification_duration_samples = int(sonification_duration * fs)

    # if sonification_duration is less than num_samples, crop the arrays
    if sonification_duration_samples < num_samples:
        pianoroll_df = pianoroll_df[pianoroll_df['start'] < sonification_duration]
        pianoroll_df.loc[pianoroll_df['end'] > sonification_duration, 'end'] = sonification_duration

    pianoroll_df['duration'] = pianoroll_df['end'] - pianoroll_df['start']

    num_samples = sonification_duration_samples
    pianoroll_sonification = np.zeros(num_samples)

    return pianoroll_df, pianoroll_sonification
