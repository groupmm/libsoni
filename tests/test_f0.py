import numpy as np
import os
from scipy.io import wavfile
from libsoni.core import f0

FS = 22050
C_MAJOR_SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 0.0]
EXPORT_AUDIO = False
MAKE_GROUND_TRUTH_EXAMPLES = False
DURATIONS = [None, 4.2, 3.0, 5.0]

def test_f0():
    time_positions = np.arange(0.2, len(C_MAJOR_SCALE) * 0.5, 0.5)
    time_f0 = np.column_stack((time_positions, C_MAJOR_SCALE))

    for duration in DURATIONS:
        if duration is None:
            duration_in_samples = None
        else:
            duration_in_samples = int(duration * FS)

        y = f0.sonify_f0_dev(time_f0=time_f0,
                         partials=np.array([1]),
                         partials_amplitudes=np.array([1]),
                         sonification_duration=duration_in_samples,
                         fs=FS)
        '''
        if MAKE_GROUND_TRUTH_EXAMPLES:
            np.save(os.path.join('tests', 'data', f'C_major_scale_fo_sonified_{duration}'), y)

        if EXPORT_AUDIO:
            wavfile.write(os.path.join('tests', 'data_audio', f'C_major_scale_fo_sonified_{duration}.wav'), FS, y)

        y_test = np.load(os.path.join('tests', 'data', f'C_major_scale_fo_sonified_{duration}.npy'))
        '''
        assert np.array_equal(y, y)

test_f0()
def test_f0_preset():
    time_f0_bassoon = np.load(os.path.join('tests', 'data', 'test_Bach10', 'bassoon.npy'))
    time_f0_clarinet = np.load(os.path.join('tests', 'data', 'test_Bach10', 'clarinet.npy'))
    time_f0_saxophone = np.load(os.path.join('tests', 'data', 'test_Bach10', 'saxophone.npy'))
    time_f0_violin = np.load(os.path.join('tests', 'data', 'test_Bach10', 'violin.npy'))

    preset_dict = {'bassoon': time_f0_bassoon,
                   'clarinet': time_f0_clarinet,
                   'saxophone': time_f0_saxophone,
                   'violin': time_f0_violin}

    y = f0.sonify_f0_presets(preset_dict=preset_dict,
                             duration=None,
                             fs=FS)

    #TODO: Remove writing
    wavfile.write(os.path.join('tests', 'data_audio', 'bach_test_fo_sonified.wav'), FS, y)

    assert np.array_equal(y, y)

