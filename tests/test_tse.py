import librosa
import numpy as np
import os
from scipy.io import wavfile
from libsoni.core import tse

FS = 22050
EXPORT_AUDIO = True
RELATIVE_OFFSETS = [0.0, 0.5, 1]
DURATIONS = [None, 0.1, 3.0, 5.0]


def test_tse_click():
    time_positions = np.arange(0, 5.3, 0.5)

    for duration in DURATIONS:
        if duration is None:
            duration_in_samples = None
        else:
            duration_in_samples = int(duration * FS)
        for relative_offset in RELATIVE_OFFSETS:
            y = tse.sonify_tse_click(time_positions=time_positions,
                                     click_pitch=69,
                                     click_duration=0.5,
                                     click_amplitude=1.0,
                                     offset_relative=relative_offset,
                                     duration=duration_in_samples,
                                     fs=FS)

            if EXPORT_AUDIO:
                wavfile.write(os.path.join('tests', 'data_audio',
                                           f'start0_time_positions_tse_clicks_sonified_{duration}_{relative_offset}.wav'),
                              FS, y)


# test_tse_click()

def test_tse_sample():
    time_positions = np.arange(0, 5.3, 0.5)

    for wav_name in os.listdir('data_audio/samples'):
        if '.wav' not in wav_name:
            continue
        sample, _ = librosa.load(os.path.join('data_audio/samples', wav_name), sr=FS)
        filename = wav_name.split('.wav')[0]
        for duration in DURATIONS:
            if duration is None:
                duration_in_samples = None
            else:
                duration_in_samples = int(duration * FS)
            for relative_offset in RELATIVE_OFFSETS:
                y = tse.sonify_tse_sample(time_positions=time_positions,
                                          sample=sample,
                                          offset_relative=relative_offset,
                                          duration=duration_in_samples)

                if EXPORT_AUDIO:
                    wavfile.write(os.path.join('tests', 'data_audio',
                                               f'{filename}_time_positions_tse_samples_sonified_{duration}_{relative_offset}.wav'),
                                  FS, y)


# test_tse_sample()


def test_tse_multiple_click():
    time_positions_eigths = np.arange(0, 6, 0.25)
    time_positions_two_four = np.arange(0.5, 6, 1)
    time_positions_one_three = np.arange(0, 6, 1)
    pitch_eigths = 81
    pitch_two_four = 69
    pitch_one_three = 45
    times_pitches = [[time_positions_eigths, pitch_eigths],
                     [time_positions_two_four, pitch_two_four],
                     [time_positions_one_three, pitch_one_three]]

    y = tse.sonify_tse_multiple_click(times_pitches=times_pitches,
                                      duration=None,
                                      click_duration=0.25,
                                      click_amplitude=1.0,
                                      offset_relative=0.0,
                                      fs=FS)

    if EXPORT_AUDIO:
        wavfile.write(os.path.join('tests', 'data_audio', 'time_positions_tse_multiple_clicks_sonified.wav'), FS, y)

test_tse_multiple_click()


def test_tse_multiple_samples():
    time_positions_eigths = np.arange(0, 6, 0.25)
    time_positions_two_four = np.arange(0.5, 6, 1)
    time_positions_one_three = np.arange(0, 6, 1)
    sample_eights, _ = librosa.load('data_audio/samples/hi-hat.wav', sr=FS)
    sample_two_four, _ = librosa.load('data_audio/samples/snare.wav', sr=FS)
    sample_one_three, _ = librosa.load('data_audio/samples/snare.wav', sr=FS)

    times_pitches = [[time_positions_eigths, pitch_eigths],
                     [time_positions_two_four, pitch_two_four],
                     [time_positions_one_three, pitch_one_three]]

    y = tse.sonify_tse_multiple_click(times_pitches=times_pitches,
                                      duration=None,
                                      click_duration=0.25,
                                      click_amplitude=1.0,
                                      offset_relative=0.0,
                                      fs=FS)

    if EXPORT_AUDIO:
        wavfile.write(os.path.join('tests', 'data_audio', 'time_positions_tse_multiple_clicks_sonified.wav'), FS, y)

test_tse_multiple_click()