import numpy as np
import os
from scipy.io import wavfile
from libsoni.core import tse

FS = 22050
EXPORT_AUDIO = True
RELATIVE_OFFSETS = [0.0, 0.5, 1]
DURATIONS = [None, 4.2, 3.0, 5.0]

def test_tse():
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
                wavfile.write(os.path.join('tests', 'data_audio', f'start0_time_positions_tse_sonified_{duration}_{relative_offset}.wav'), FS, y)

test_tse()