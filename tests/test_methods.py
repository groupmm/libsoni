import numpy as np
import soundfile as sf

from libsoni.core.methods import generate_click, generate_shepard_tone, generate_sinusoid
from libsoni.util.utils import pitch_to_frequency

DURATIONS = [0.2, 0.5, 1.0]
FADE = [0.05, 0.1]
PITCHES = [60, 69]
Fs = 22050


def test_click():
    for duration in DURATIONS:
        for pitch in PITCHES:
            y = generate_click(pitch=pitch,
                               click_fading_duration=duration)

            ref, _ = sf.read(f'tests/data/click_{pitch}_{(int(Fs * duration))}.wav')

            assert len(y) == len(ref), 'Length of the generated sonification does not match with the reference!'
            assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)


def test_sinusoid():
    for duration in DURATIONS:
        for pitch in PITCHES:
            for fading_duration in FADE:
                freq = pitch_to_frequency(pitch=pitch)
                y = generate_sinusoid(frequency=freq,
                                      duration=duration,
                                      fading_duration=fading_duration)
                dur_samples = int(Fs * duration)
                fade_samples = int(Fs * fading_duration)
                ref, _ = sf.read(f'tests/data/sin_{pitch}_{dur_samples}_{fade_samples}.wav')
                assert len(y) == len(ref), 'Length of the generated sonification does not match with the reference!'
                assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)


def test_shepard_tone():
    for duration in DURATIONS:
        for pitch in PITCHES:
            for fading_duration in FADE:
                pitch_class = pitch % 12
                y = generate_shepard_tone(pitch_class=pitch_class,
                                          duration=duration,
                                          fading_duration=fading_duration)
                dur_samples = int(Fs * duration)
                fade_samples = int(Fs * fading_duration)
                ref, _ = sf.read(f'tests/data/shepard_{pitch_class}_{dur_samples}_{fade_samples}.wav')
                assert len(y) == len(ref), 'Length of the generated sonification does not match with the reference!'
                assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

