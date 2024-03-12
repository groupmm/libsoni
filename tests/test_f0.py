import numpy as np
import soundfile as sf

from libsoni.core import f0

Fs = 22050
C_MAJOR_SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 0.0]
DURATIONS = [None, 3.0, 5.0]
PARTIALS = [np.array([1]), np.array([1, 2, 3])]
PARTIALS_AMPLITUDES = [np.array([1]), np.array([1, 0.5, 0.25])]


def test_f0():
    time_positions = np.arange(0.2, len(C_MAJOR_SCALE) * 0.5, 0.5)
    time_f0 = np.column_stack((time_positions, C_MAJOR_SCALE))

    for duration in DURATIONS:
        for par_idx, partials in enumerate(PARTIALS):
            if duration is None:
                duration_in_samples = None
            else:
                duration_in_samples = int(duration * Fs)

            y = f0.sonify_f0(time_f0=time_f0,
                             partials=partials,
                             partials_amplitudes=PARTIALS_AMPLITUDES[par_idx],
                             sonification_duration=duration_in_samples,
                             fs=Fs)

            ref, _ = sf.read(f'tests/data/f0_{duration_in_samples}_{par_idx}.wav')
            assert len(y) == len(ref), 'Length of the generated sonification does not match with the reference!'
            assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)
