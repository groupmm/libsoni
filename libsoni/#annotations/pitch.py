import numpy as np
import pandas as pd
from libsoni.util.utils import click, add_to_sonification, generate_additive_synthesized_tone, \
    generate_fm_synthesized_tone, envelop_signal
from typing import List


def sonify_pitch_annotation(path_to_csv: str,
                            sonification_method: str = 'click',
                            instrument_mask: list = None,
                            voice_mask: list = None,

                            # parameters additive synthesis
                            frequency_ratios: List = [1],
                            frequency_ratios_amp: list = [1],
                            frequency_ratios_phase_offsets: list = [0],

                            # parameters fm synthesis
                            modulation_frequency: float = 0,
                            modulation_index: float = 0,

                            # parameters ADSR
                            use_ADSR: bool = False,
                            attack_time: float = 0,
                            decay_time: float = 0,
                            sustain_level: float = 0,
                            release_time: float = 0,

                            # general parameters
                            fade_dur: float = 0.01,
                            tuning_frequency: float = 440,
                            fs: int = 44100):
    """
        This function sonifies the entries of a pitch annotation in .csv format.
        (see /docs/annotation_conventions.txt for more information)

        Parameters
        ----------
        path_to_csv : str
            path to annotation file
        sonification_method : str
            method of sonification (default: 'click')
        instrument_mask: list
            used to only sonify pitch events of chosen instruments
        voice_mask: list
            used to only sonify pitch events of chosen voices
        frequency_ratios: np.ndarray
            frequencies to use for additive synthesis
        frequency_ratios_amp: np.ndarray
            amplitude ratios of frequencies to use for additive synthesis
        frequency_ratios_phase_offsets: np.ndarray
            phase offsets of frequencies to use for additive synthesis
        modulation_frequency: float
            modulation frequency to use for frequency modulation synthesis
        modulation_index: float
            amount of frequency modulation
        use_ADSR: bool
            use ADSR - envelope
        attack_time: float
            attack time of ADSR - envelope
        decay_time: float
            decay time of ADSR - envelope
        sustain_level: float
            sustain level of ADSR - envelope
        release_time: float
            release time of ADSR - envelope
        fade_dur: float
            duration of fading in and out
        tuning_frequency: float
            tuning frequency
        fs: int
            sampling rate

        Returns
        ----------
        y: array-like
            sonification
        """

    # read annotation file
    pitch_annotation_df = pd.read_csv(path_to_csv, delimiter=';')

    # make copy of annotation file with columns 'start', 'end' and 'pitch' only.
    pitch_annotation_shortened_df = pitch_annotation_df[['start', 'end', 'pitch']]

    # if use_ADSR is set to True, ignore fade_dur parameter
    if use_ADSR:
        fade_dur = 0

    # create empty array according to the time bounds given by the annotation file
    y = np.zeros(np.ceil(max(pitch_annotation_df.end.unique()) * fs).astype(int))

    # iterate pitch events of the annotation file and insert corresponding click signals at the
    # corresponding temporal positions
    for i, r in pitch_annotation_shortened_df.iterrows():
        start, end, pitch = r

        # check if 'instrument' information is given in annotation file
        if 'instrument' in list(pitch_annotation_df.columns):

            # check if instrument of pitch event is listed in instrument_mask else skip pitch event
            if instrument_mask is not None and pitch_annotation_df.iloc[i]['instrument'] in instrument_mask:
                pass
            else:
                continue

        # check if 'voice' information is given in annotation file
        elif 'voice' in list(pitch_annotation_df.columns):
            if voice_mask is not None and pitch_annotation_df.iloc[i]['voice'] in voice_mask:
                pass
                # skip if voice of pitch event is omitted
                continue

        else:
            pass

        if 'velocity' in list(pitch_annotation_df.columns):
            amplitude = pitch_annotation_df.iloc[i]['velocity'] / 127
        else:
            amplitude = 1

        duration = end - start
        if duration <= 0:
            continue
        if sonification_method == 'click':
            pitch_sonification = click(pitch=pitch, amplitude=amplitude, duration=duration, fs=fs)
        elif sonification_method == 'additive_synthesis':
            pitch_sonification = generate_additive_synthesized_tone(pitch=pitch,
                                                                    frequency_ratios=frequency_ratios,
                                                                    frequency_ratios_amp=frequency_ratios_amp,
                                                                    frequency_ratios_phase_offsets=frequency_ratios_phase_offsets,
                                                                    amp=amplitude,
                                                                    dur=duration,
                                                                    fs=fs,
                                                                    f_tuning=tuning_frequency,
                                                                    fade_dur=fade_dur)
        elif sonification_method == 'fm_synthesis':
            pitch_sonification = generate_fm_synthesized_tone(pitch=pitch,
                                                              modulation_frequency=modulation_frequency,
                                                              modulation_index=modulation_index,
                                                              amp=amplitude,
                                                              dur=duration,
                                                              fs=fs,
                                                              f_tuning=tuning_frequency,
                                                              fade_dur=fade_dur)
        if use_ADSR:
            pitch_sonification = envelop_signal(pitch_sonification, attack_time=attack_time, decay_time=decay_time,
                                                sustain_level=sustain_level, release_time=release_time)
        y = add_to_sonification(sonification=y, sonification_for_event=pitch_sonification, start=start, fs=fs)

    return y
