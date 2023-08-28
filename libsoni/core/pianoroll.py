import numpy as np
import pandas as pd

from libsoni.util.utils import generate_click, generate_tone_additive_synthesis, format_df, warp_sample


def sonify_pianoroll_sample(pianoroll_df: pd.DataFrame,
                            sample: np.ndarray = None,
                            reference_pitch: int = 69,
                            duration: int = None,
                            fs: int = 22050) -> np.ndarray:
    """This function sonifies a pianoroll representation containing pitch events described by start, duration or end
        and the corresponding pitch with pitch-shifted and time-warped versions of a sample.

        Parameters
        ----------
        pianoroll_df: pd.DataFrame
            Data Frame containing pitch events.
        sample: np.ndarray
            Sample
        sample_pitch: int, default = 69
            Pitch of the Sample
        duration: float, default = None
            Duration of the output waveform, given in samples.
        fs: int, default = 22050
            Sampling rate

        Returns
        -------
        pianoroll_sonification: np.ndarray
            Sonified waveform in form of a 1D Numpy array.
        """

    pianoroll_df = format_df(pianoroll_df)

    shorter_duration = False

    num_samples = int(pianoroll_df['end'].max() * fs)

    if duration is not None:

        duration_in_sec = duration / fs

        if duration == num_samples:
            pass

        elif duration < num_samples:
            pianoroll_df = pianoroll_df[pianoroll_df['start'] < duration]
            pianoroll_df['end'] = pianoroll_df[pianoroll_df['end'] > duration] = duration
            pianoroll_df['duration'] = pianoroll_df['end'] - pianoroll_df['start']
        num_samples = duration

    pianoroll_sonification = np.zeros(num_samples)

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        amplitude = r['velocity'] if 'velocity' in r else 1.0

        warped_sample = warp_sample(sample=sample,
                                    reference_pitch=reference_pitch,
                                    target_pitch=r['pitch'],
                                    target_duration_sec=r['duration'],
                                    fs=fs)

        pianoroll_sonification[start_samples:start_samples + len(warped_sample)] += warped_sample

    return pianoroll_sonification

def sonify_pianoroll_clicks(pianoroll_df: pd.DataFrame,
                            tuning_frequency: float = 440.0,
                            duration: int = None,
                            fs: int = 22050) -> np.ndarray:
    """This function sonifies a pianoroll representation containing pitch events described by start, duration or end
    and the corresponding pitch with coloured clicks.

    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Data Frame containing pitch events.
    tuning_frequency: float, default = 440.0
        Tuning Frequency, given in Hertz
    duration: float, default = None
        Duration of the output waveform, given in samples.
    fs: int, default = 22050
        Sampling rate

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """
    pianoroll_df = format_df(pianoroll_df)

    shorter_duration = False

    num_samples = int(pianoroll_df['end'].max() * fs)

    if duration is not None:

        duration_in_sec = duration / fs

        if duration == num_samples:
            pass

        elif duration < num_samples:
            pianoroll_df = pianoroll_df[pianoroll_df['start'] < duration]
            pianoroll_df['end'] = pianoroll_df[pianoroll_df['end'] > duration] = duration
            pianoroll_df['duration'] = pianoroll_df['end'] - pianoroll_df['start']
        num_samples = duration

    pianoroll_sonification = np.zeros(num_samples)

    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        amplitude = r['velocity'] if 'velocity' in r else 1.0

        pianoroll_sonification[start_samples:start_samples + duration_samples] += generate_click(pitch=r['pitch'],
                                                                                                 amplitude=amplitude,
                                                                                                 duration=r['duration'],
                                                                                                 fs=fs,
                                                                                                 tuning_frequency=
                                                                                                 tuning_frequency)

    return pianoroll_sonification


def sonify_pianoroll_additive_synthesis(pianoroll_df: pd.DataFrame,
                                        partials: np.ndarray = np.array([1]),
                                        partials_amplitudes: np.ndarray = None,
                                        partials_phase_offsets=None,
                                        tuning_frequency: float = 440.0,
                                        duration: int = None,
                                        fs: int = 22050) -> np.ndarray:
    # TODO: conventions as in sonify_f0 (partials..)
    pianoroll_df = format_df(pianoroll_df)
    shorter_duration = False
    num_samples = int(pianoroll_df['end'].max() * fs)
    if duration is not None:
        duration_in_sec = duration / fs
        # if duration equals num_samples, do nothing
        if duration == num_samples:
            pass
        # if duration is less than num_samples, crop the arrays
        elif duration < num_samples:
            pianoroll_df = pianoroll_df[pianoroll_df['start'] < duration]
            pianoroll_df['end'] = pianoroll_df[pianoroll_df['end'] > duration] = duration
            pianoroll_df['duration'] = pianoroll_df['end'] - pianoroll_df['start']
        num_samples = duration
    pianoroll_sonification = np.zeros(num_samples)
    for i, r in pianoroll_df.iterrows():
        start_samples = int(r['start'] * fs)
        duration_samples = int(r['duration'] * fs)
        # TODO: check velocity values -> right scaling
        amplitude = r['velocity'] if 'velocity' in r else 1.0

        pianoroll_sonification[start_samples:start_samples + duration_samples] += \
            generate_tone_additive_synthesis(pitch=r['pitch'],
                                             partials=partials,
                                             partials_amplitudes=partials_amplitudes,
                                             partials_phase_offsets=partials_phase_offsets,
                                             gain=amplitude,
                                             duration_sec=r['duration'],
                                             fs=fs,
                                             f_tuning=tuning_frequency)


    return pianoroll_sonification


def sonify_pianoroll_frequency_modulation_synthesis():
    return


def sonify_pianoroll_etc():
    return



