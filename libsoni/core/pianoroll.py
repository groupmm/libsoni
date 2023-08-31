import libfmp.b
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from libsoni.util.utils import format_df, warp_sample, normalize_signal
from libsoni.core.methods import generate_click, generate_tone_additive_synthesis


def sonify_pianoroll_additive_synthesis(pianoroll_df: pd.DataFrame,
                                        partials: np.ndarray = np.array([1]),
                                        partials_amplitudes: np.ndarray = None,
                                        partials_phase_offsets=None,
                                        tuning_frequency: float = 440.0,
                                        sonification_duration: int = None,
                                        normalize: bool = True,
                                        fs: int = 22050) -> np.ndarray:
    """
    TODO: Let's discuss about the input data type
    Parameters
    ----------
    pianoroll_df: pd.DataFrame
        Dataframe
    partials
    partials_amplitudes
    partials_phase_offsets
    tuning_frequency
    sonification_duration
    normalize
    fs

    Returns
    -------

    """
    pianoroll_df = format_df(pianoroll_df)
    num_samples = int(pianoroll_df['end'].max() * fs)
    if sonification_duration is not None:
        # if sonification_duration equals num_samples, do nothing
        if sonification_duration == num_samples:
            pass
        # if sonification_duration is less than num_samples, crop the arrays
        elif sonification_duration < num_samples:
            pianoroll_df = pianoroll_df[pianoroll_df['start'] < sonification_duration]
            pianoroll_df['end'] = pianoroll_df[pianoroll_df['end'] > sonification_duration] = sonification_duration
            pianoroll_df['duration'] = pianoroll_df['end'] - pianoroll_df['start']
        num_samples = sonification_duration

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

    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def sonify_pianoroll_clicks(time_positions: np.ndarray = None,
                            pitches: np.ndarray = None,
                            fading_durations: np.ndarray = None,
                            velocities: np.ndarray = None,
                            tuning_frequency: float = 440.0,
                            sonification_duration: int = None,
                            normalize: bool = True,
                            fs: int = 22050) -> np.ndarray:
    """This function sonifies a pianoroll representation containing pitch events described by start,
    sonification_duration or end and the corresponding pitch with coloured clicks.

    Parameters
    ----------
    time_positions: np.ndarray
        1D-array of starting time positions of the pitches in the piano roll
    pitches: np.ndarray
        1D-array of MIDI pitches in the piano roll
    fading_durations: np.ndarray
        1D-array of fading durations of the clicks. If None, 0.25 seconds are used for each click.
    velocities: np.ndarray
        1D-array of key velocities. If None, 1.0 is used for each click.
    tuning_frequency: float, default = 440.0
        Tuning frequency, given in Hertz
    sonification_duration: float, default = None
        Duration of the output waveform, given in samples.
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate

    Returns
    -------
    pianoroll_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """
    num_samples = int(time_positions[-1] * fs) + int(fading_durations[-1] * fs) if fading_durations is not None \
        else int(time_positions[-1] * fs) + int(0.25 * fs)

    if sonification_duration is not None:
        if sonification_duration == num_samples:
            pass

        elif sonification_duration < num_samples:
            duration_in_sec = sonification_duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]
            len_cropped_indices = len(time_positions)
            pitches = pitches[:len_cropped_indices] if pitches is not None else pitches
            fading_durations = fading_durations[:len_cropped_indices] if pitches is not None else\
                fading_durations
            velocities = velocities[:len_cropped_indices] if velocities is not None else velocities
        num_samples = sonification_duration

    else:
        sonification_duration = num_samples

    pianoroll_sonification = np.zeros(num_samples)

    for idx in range(len(time_positions)):
        pitch = 69 if pitches is None else pitches[idx]
        fading_duration = 0.25 if fading_durations is None else fading_durations[idx]
        amplitude = 1.0 if velocities is None else velocities[idx]
        start_samples = int(time_positions[idx] * fs)
        end_samples = start_samples + int(fading_duration * fs)

        click = generate_click(pitch=pitch,
                               amplitude=amplitude,
                               fading_duration=fading_duration,
                               fs=fs,
                               tuning_frequency=tuning_frequency)

        if end_samples < sonification_duration:
            pianoroll_sonification[start_samples:end_samples] += click
        else:
            pianoroll_sonification[start_samples:sonification_duration] += click[:sonification_duration-start_samples]

    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def sonify_pianoroll_sample(pianoroll_df: pd.DataFrame,
                            sample: np.ndarray = None,
                            reference_pitch: int = 69,
                            duration: int = None,
                            normalize: bool = True,
                            fs: int = 22050) -> np.ndarray:
    """This function sonifies a pianoroll representation containing pitch events described by start, sonification_duration or end
        and the corresponding pitch with pitch-shifted and time-warped versions of a sample.

        Parameters
        ----------
        pianoroll_df: pd.DataFrame
            Data Frame containing pitch events.
        sample: np.ndarray
            Sample
        reference_pitch: int, default = 69
            Pitch of the Sample
        duration: float, default = None
            Duration of the output waveform, given in samples.
        normalize: bool, default = True
            Decides, if output signal is normalized to [-1,1].
        fs: int, default = 22050
            Sampling rate

        Returns
        -------
        pianoroll_sonification: np.ndarray
            Sonified waveform in form of a 1D Numpy array.
        """
    pianoroll_df = format_df(pianoroll_df)
    num_samples = int(pianoroll_df['end'].max() * fs)

    if duration is not None:
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
        warped_sample = warp_sample(sample=sample,
                                    reference_pitch=reference_pitch,
                                    target_pitch=r['pitch'],
                                    target_duration_sec=r['duration'],
                                    fs=fs)

        pianoroll_sonification[start_samples:start_samples + len(warped_sample)] += warped_sample

    pianoroll_sonification = normalize_signal(pianoroll_sonification) if normalize else pianoroll_sonification

    return pianoroll_sonification


def visualize_pianoroll(df, xlabel='Time (seconds)', ylabel='Pitch', colors='FMP_1', velocity_alpha=False,
                         figsize=(12, 4), ax=None, dpi=72):
    """Plot a pianoroll visualization, inspired from FMP Notebook C1/C1S2_CSV.ipynb

    Args:
        score: List of note events
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = 'Pitch')
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications,
            4. dict that assigns labels  to colors (Default value = 'FMP_1')
        velocity_alpha: Use the velocity value for the alpha value of the corresponding rectangle
            (Default value = False)
        figsize: Width, height in inches (Default value = (12)
        ax: The Axes instance to plot on (Default value = None)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes
    """
    df = format_df(df)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    labels_set = sorted(df['label'].unique())
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)

    pitch_min = df['pitch'].min()
    pitch_max = df['pitch'].max()
    time_min = df['start'].min()
    time_max = df['end'].max()

    for i, r in df.iterrows():
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((r['start'], r['pitch'] - 0.5), r['duration'], 1, linewidth=1,
                                 edgecolor='k', facecolor=colors[r['label']], alpha=r['velocity'])
        ax.add_patch(rect)

    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def sonify_pianoroll_frequency_modulation_synthesis():
    return


def sonify_pianoroll_etc():
    return





