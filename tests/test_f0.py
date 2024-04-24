import numpy as np
import soundfile as sf
from unittest import TestCase


from libsoni.core import f0


class TestF0(TestCase):
    def setUp(self) -> None:
        c_major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 0.0]
        time_positions = np.arange(0.2, len(c_major_scale) * 0.5, 0.5)
        self.fs = 22050
        self.durations = [int(3.0*self.fs), int(5.0*self.fs)]
        self.partials = [np.array([1]), np.array([1, 2, 3])]
        self.partials_amplitudes = [np.array([1]), np.array([1, 0.5, 0.25])]
        self.time_f0 = np.column_stack((time_positions, c_major_scale))

    def test_input_types(self) -> None:
        [self.assertIsInstance(duration, int) for duration in self.durations]
        [self.assertIsInstance(partials, np.ndarray) for partials in self.partials]
        [self.assertIsInstance(partials_amplitude, np.ndarray) for partials_amplitude in self.partials_amplitudes]
        self.assertIsInstance(self.fs, int)
        self.assertIsInstance(self.time_f0, np.ndarray)

    def test_input_shape(self) -> None:
        with self.assertRaises(IndexError) as context:
            _ = f0.sonify_f0(time_f0=np.zeros(1))
        self.assertEqual(str(context.exception), 'time_f0 must be a numpy array of size [N, 2]')

        with self.assertRaises(IndexError) as context:
            _ = f0.sonify_f0(time_f0=np.zeros((3, 3)))
        self.assertEqual(str(context.exception), 'time_f0 must be a numpy array of size [N, 2]')

    def test_invalid_partial_sizes(self):
        with self.assertRaises(ValueError) as context:
            _ = f0.sonify_f0(time_f0=self.time_f0,
                             partials=self.partials[0],
                             partials_amplitudes=self.partials_amplitudes[1],
                             sonification_duration=self.durations[0],
                             fs=self.fs)

        self.assertEqual(str(context.exception), 'Partials, Partials_amplitudes and Partials_phase_offsets must be '
                                                 'of equal length.')

    def test_sonification(self) -> None:
        for duration in self.durations:
            for par_idx, partials in enumerate(self.partials):
                y = f0.sonify_f0(time_f0=self.time_f0,
                                 partials=self.partials[par_idx],
                                 partials_amplitudes=self.partials_amplitudes[par_idx],
                                 sonification_duration=duration,
                                 fs=self.fs)

                ref, _ = sf.read(f'tests/data/f0_{duration}_{par_idx}.wav')
                self.assertEqual(len(y), len(ref), msg='Length of the generated sonification '
                                                       'does not match with the reference!')
                assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)
