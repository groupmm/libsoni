import numpy as np
import pandas as pd

from libsoni.util.utils import generate_click, generate_additive_synthesized_tone


def sonify_pianoroll_clicks(pianoroll_df: pd.DataFrame,
                            tuning_frequency: float = 440.0,
                            duration: int = None,
                            fs: int = 22050) -> np.ndarray:
    pianoroll_df = __format_df(pianoroll_df)

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
        amplitude = r['velocity'] if 'velocity' in r else 1.0

        pianoroll_sonification[start_samples:start_samples+duration_samples] += generate_click(pitch=r['pitch'],
                                                                            amplitude=amplitude,
                                                                            duration=r['duration'],
                                                                            fs=fs,
                                                                            tuning_frequency=tuning_frequency)

    return pianoroll_sonification


def sonify_pianoroll_additive_synthesis(pianoroll_df: pd.DataFrame,
                                        frequency_ratios: np.ndarray = np.array([1]),
                                        frequency_ratios_amp: np.ndarray = np.array([1]),
                                        frequency_ratios_phase_offsets= np.array([1]),
                                        tuning_frequency: float = 440.0,
                                        duration: int = None,
                                        fs: int = 22050) -> np.ndarray:
    # TODO: conventions as in sonify_f0 (partials..)
    pianoroll_df = __format_df(pianoroll_df)
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
        amplitude = r['velocity'] if 'velocity' in r else 1.0
        pianoroll_sonification[start_samples:start_samples + duration_samples] += \
            generate_additive_synthesized_tone(pitch=r['pitch'],
                                               frequency_ratios=frequency_ratios,
                                               frequency_ratios_amp=frequency_ratios_amp,
                                               frequency_ratios_phase_offsets=frequency_ratios_phase_offsets,
                                               amp=amplitude,
                                               dur=r['duration'],
                                               fs=fs,
                                               f_tuning=tuning_frequency)


    return pianoroll_sonification



def sonify_pianoroll_frequency_modulation_synthesis():
    return


def sonify_pianoroll_etc():
    return


def __format_df(df):
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
