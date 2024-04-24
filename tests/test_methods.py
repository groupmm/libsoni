import numpy as np
import soundfile as sf
from unittest import TestCase

from libsoni.core.methods import generate_click, generate_shepard_tone, generate_sinusoid,\
    generate_tone_instantaneous_phase
from libsoni.util.utils import pitch_to_frequency


class TestMethods(TestCase):
    def setUp(self) -> None:
        self.frequency_vector = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 0.0])
        self.pitches = [60, 69]
        self.fade_vals = [0.05, 0.1]
        self.fs = 22050
        self.durations = [int(0.2*self.fs), int(0.5*self.fs), int(1.0*self.fs)]
        self.partials = [np.array([1]), np.array([1, 2, 3])]
        self.partials_amplitudes = [np.array([1]), np.array([1, 0.5, 0.25])]

    def test_input_types(self) -> None:
        [self.assertIsInstance(duration, int) for duration in self.durations]
        [self.assertIsInstance(freq, float) for freq in self.frequency_vector]
        [self.assertIsInstance(partials, np.ndarray) for partials in self.partials]
        [self.assertIsInstance(partials_amplitude, np.ndarray) for partials_amplitude in self.partials_amplitudes]
        self.assertIsInstance(self.fs, int)

    def test_invalid_partial_sizes(self) -> None:
        with self.assertRaises(ValueError) as context:
            _ = generate_tone_instantaneous_phase(self.frequency_vector,
                                                  partials=self.partials[0],
                                                  partials_amplitudes=self.partials_amplitudes[1],
                                                  fs=self.fs)

        self.assertEqual(str(context.exception), 'Partials, Partials_amplitudes and Partials_phase_offsets must be '
                                                 'of equal length.')

    def test_click(self) -> None:
        for duration in self.durations:
            for pitch in self.pitches:
                for fade_val in self.fade_vals:
                    freq = pitch_to_frequency(pitch=pitch)
                    y = generate_sinusoid(frequency=freq,
                                          duration=duration/self.fs,
                                          fading_duration=fade_val)
                    fade_samples = int(self.fs * fade_val)
                    ref, _ = sf.read(f'tests/data/sin_{pitch}_{duration}_{fade_samples}.wav')
                    self.assertEqual(len(y), len(ref), msg='Length of the generated sonification '
                                                           'does not match with the reference!')
                    assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

    def test_shepard_tone(self) -> None:
        for duration in self.durations:
            for pitch in self.pitches:
                for fade_val in self.fade_vals:
                    pitch_class = pitch % 12
                    y = generate_shepard_tone(pitch_class=pitch_class,
                                              duration=duration/self.fs,
                                              fading_duration=fade_val)
                    fade_samples = int(self.fs * fade_val)
                    ref, _ = sf.read(f'tests/data/shepard_{pitch_class}_{duration}_{fade_samples}.wav')
                    self.assertEqual(len(y), len(ref), msg='Length of the generated sonification '
                                                           'does not match with the reference!')
                    assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)
