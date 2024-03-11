---
title: 'libsoni: A Python Package for Music Sonification'
tags:
  - Python
  - Music information retrieval
  - Music sonification
authors:
  - name: Yigitcan Özer
    orcid: 0000-0003-2235-8655
    affiliation: 1
  - name: Leo Brütting
    affiliation: 1
  - name: Simon Schwär
    orcid: 0000-0001-5780-557X
    affiliation: 1
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    affiliation: 1
affiliations:
 - name: International Audio Laboratories Erlangen
   index: 1
date: 12 March 2024
bibliography: references.bib
link-citations: yes
---

# Summary
Music Information Retrieval (MIR) stands as a dedicated research field focused on advancing methodologies and tools for 
organizing, analyzing, retrieving, and generating data related to music. Key tasks within MIR include beat tracking, 
structural analysis, chord recognition, melody extraction, and source separation, just to name a few. These tasks
involve extracting musically relevant information from audio recordings, typically accomplished by transforming music 
signals into feature representations such as spectrograms, chromagrams, or tempograms [@Mueller15_FMP_SPRINGER].
Furthermore, musically relevant annotations such as beats, chords, keys, or structure boundaries become
indispensable for training and evaluating MIR approaches.

When evaluating and enhancing MIR systems, it is crucial to thoroughly examine the properties of feature representations
and annotations to gain a deeper understanding of algorithmic behavior and the underlying data. In the musical context,
alongside conventional data visualization techniques, data sonification techniques are emerging as a promising avenue 
for providing auditory feedback on extracted features or annotated information. This is particularly advantageous given 
the finely tuned human perception to subtle variations in frequency and timing within the musical domain.

This paper introduces *libsoni*, an open-source Python toolbox tailored for the sonification of music annotations 
and feature representations. By employing explicit and easy-to-understand sound synthesis techniques, libsoni offers 
functionalities for generating and triggering sound events, enabling the sonification of spectral, harmonic, tonal, 
melodic, and rhythmic aspects. Unlike existing software libraries focused on  creative applications of sound generation,
libsoni is designed to meet the specific needs of MIR researchers and educators. It aims to simplify the process of 
music exploration, promoting a more intuitive and efficient approach to data analysis by enabling users to interact with 
their data in acoustically meaningful ways. As a result, libsoni not only improves the analytical capabilities of music
scientists but also opens up new avenues for innovative music analysis and discovery.

Furthermore, libsoni provides well-documented and stand-alone functions covering all essential building blocks crucial 
for both sound generation and sonification, enabling users to efficiently apply and easily extend the methods. 
Additionally, the toolbox includes educational Jupyter notebooks with illustrative code examples demonstrating the 
application of sonification and visualization methods to deepen understanding within specific MIR scenarios.


# Statement of Need
Music data, characterized by attributes such as pitch, melody, harmony, rhythm, structure, and timbre, is inherently 
intricate. Visualizations are crucial in deciphering this complexity by presenting music representations graphically,
enabling researchers to identify patterns, trends, and relationships not immediately evident in raw data.
For instance, visualizing time-dependent two-dimensional feature representations such as spectrograms (time--frequency),
chromagrams (time--chroma), or tempograms (time--tempo) enhances comprehension of signal processing concepts 
insights into the musical and acoustic properties of audio signals. Moreover, the combined visualization of extracted 
features and reference annotations facilitates detailed examination of algorithmic approaches at a granular level. 
These qualitative assessments, alongside quantitative metrics, are essential for comprehending the strengths, 
weaknesses, and assumptions underlying music processing algorithms.
The MIR community has developed numerous toolboxes, such as essentia [@BogdanovWGGHMRSZS13_essentia_ISMIR],
madmom [@BoeckKSKW16_madmom_ACM-MM], Chroma Toolbox [@MuellerEwert11_ChromaToolbox_ISMIR], Tempogram Toolbox 
[@GroscheM11_TempogramToolbox_ISMIR-lateBreaking], Sync Toolbox [@MuellerOKPD21_SyncToolbox_JOSS], Marsyas 
[@Tzanetakis09_MARSYAS_ACM-MM], or the  MIRtoolbox [@LartillotT07_MirToolbox_ISMIR], offering modular code for music 
signal processing and analysis, many of which also include data visualization methods.
Notably, the two Python packages librosa [@McFeeRLEMBN15_librosa_Python] and libfmp [@MuellerZ21_libfmp_JOSS] aim to
lower the barrier to entry for MIR research by providing accessible code alongside visualization functions,
bridging the gap between education and research.

As an alternative or addition to visualizing data, one can employ data sonification techniques to produce acoustic 
feedback on extracted or annotated information [@KramerEtAl99SonificAR]. This is especially important in music,
where humans excel at detecting even minor variations in the frequency and timing of sound events.
For instance, people can readily perceive irregularities and subtle changes in rhythm when they listen to a pulse 
track converted into a sequence of click sounds. This ability is particularly valuable for MIR tasks such as beat
tracking and rhythm analysis. Moreover, transforming frequency trajectories into sound using sinusoidal models can 
offer insights for tasks like estimating melody or separating singing voices. Furthermore, an auditory representation
of a chromagram provides listeners with an understanding of the harmony-related tonal information contained in an 
audio signal. Therefore, by converting data into sound, sonification can reveal subtle audible details in music 
that may not be immediately apparent within visual representations.

In the MIR context, sonification methods have been employed to provide deeper insights into various music annotations 
and feature representations. For instance, the Python package librosa [@McFeeRLEMBN15_librosa_Python] offers a function 
\texttt{librosa.clicks} that generates an audio signal with click sounds positioned at specified times, with options 
to adjust the frequency and duration of the clicks. Additionally, the Python toolbox libf0 [@RosenzweigSM22_libf0_ISMIR-LBD] 
provides a function (\texttt{libf0.utils.sonify\_trajectory\_with\_sinusoid}) for sonifying F0 trajectories using sinusoids.
Moreover, the Python package libfmp~\citep{MuellerZ21_libfmp_JOSS} includes a function 
(\texttt{libfmp.b.sonify\_chromagram\_with\_signal}) for sonifying time--chroma representations.
Testing these methods, our experiments have revealed that current implementations frequently rely on inefficient
event-based looping, resulting in excessively long runtimes. For instance, generating a click soundtrack for beat 
annotations of 10-minute recordings can require \meinard{impractically long processing times}.

In our Python toolbox, libsoni, we offer implementations of various sonification methods, including those 
mentioned above. These implementations feature a coherent API and are based on straightforward methods that are 
transparent and easy to understand. By utilizing efficient matrix-based implementations, the need for looping is 
avoided, making them more efficient. Additionally, libsoni includes all essential components for sound synthesis, 
operating as a standalone tool that can be easily extended and customized. The methods in libsoni enable
interactivity, allowing for data manipulation and sonification, as well as the ability to alter feature 
extraction or sonification techniques. While real-time capabilities are not currently included in libsoni,
this could be a potential future extension. Hence, libsoni may not only be beneficial for MIR researchers but also for
educators, students, composers, sound designers, and individuals exploring new musical concepts.


## Chromagram Representations (libsoni.chroma)
Humans perceive pitch in a periodic manner, meaning that pitches separated by an octave are perceived as having a 
similar quality or acoustic color, known as chroma. This concept motivates the use of time--chroma representations 
or chromagrams, where pitch bands that differ spectrally by one or several octaves are combined to form a single chroma
band~\citep{MuellerEwert11_ChromaToolbox_ISMIR}. These representations capture tonal information related to harmony and 
melody while exhibiting a high degree of invariance with respect to timbre and instrumentation. 
Chromagrams are widely used in MIR research for various tasks, including chord recognition and structure analysis.
The libsoni.chroma module provides sonification methods for chromagrams based on Shepard tones. 
These tones are weighted combinations of sinusoids separated by octaves and serve as acoustic counterparts to 
chroma values. The functions offered by libsoni enable the generation of various Shepard tone variants and can be 
applied to symbolic representations (such as piano roll representations or chord annotations) or to chroma features 
extracted from music recordings. This facilitates deeper insights for listeners into chord recognition results or the 
harmony-related tonal information contained within an audio signal.


## Spectrogram Representations (libsoni.spectrogram)
Similar to chromagrams, pitch-based feature representations can be derived directly from music recordings using 
transforms such as the constant-Q-transform (CQT), see\citep{SchoerkhuberK10_ConstantQTransform_SMC}. 
These representations are a special type of log-frequency spectrograms, where the frequency axis is logarithmically 
spaced to form a pitch-based axis. More generally, in audio signal processing, there exists a multitude of different
time--frequency representations. For example, classic spectrograms have a linear frequency axis, usually computed via
the short-time Fourier transform (STFT). Additionally, mel-frequency spectrograms utilize the mel scale, 
which approximates the human auditory system's response to different frequencies. The Spectrogram module of libsoni
is designed to sonify various types of spectrograms with frequency axes spaced according to linear, logarithmic,
or mel scales. Essentially, each point on the scale corresponds to a specific center frequency,
meaning that each row of the spectrogram represents the energy profile of a specific frequency over time. 
Our sonification approach generates sinusoids for each center frequency value with time-varying amplitude values,
in accordance with the provided energy profiles, and then superimposes all these sinusoids. Transforming
spectrogram-like representations into an auditory experience, our sonification approach allows for a more 
intuitive understanding of the frequency and energy characteristics within a given music recording.


# Design Choices
When designing the Python toolbox libsoni, we had several objectives in mind. Firstly, we aimed to maintain close 
connections with existing sonification methods provided in in librosa [@McFeeRLEMBN15_librosa_Python] and 
libfmp [@MuellerZ21_libfmp_JOSS]. Secondly, we re-implemented and included all necessary components 
(e.g., sound generators based on sinusoidal models and click sounds), even though similar basic functionality is 
available in other Python packages such as librosa and libfmp. By doing so, libsoni offers a coherent API along with 
convenient but easily modifiable parameter presets. Additionally, the implementations are more efficient than previous 
software. Thirdly, we adopted many design principles suggested by librosa [@McFeeRLEMBN15_librosa_Python]  
and detailed in [@McFeeKCSBB19_OpenSourcePractices_IEEE-SPM] to lower the entry barrier for students and 
researchers who may not be coding experts. This includes maintaining an explicit and straightforward programming 
style with a flat, functional hierarchy to facilitate ease of use and comprehension. The source code for
libsoni, along with comprehensive API documentation [^1], is publicly accessible through a dedicated GitHub
repository [^2]. We showcase all components, including introductions to MIR scenarios, illustrations, and sound examples
via Jupyter notebooks.  Finally, we have included the toolbox in the Python Package Index (PyPI), enabling
installation with the standard Python package manager, pip [^3].

[^1]: <https://groupmm.github.io/libsoni>
[^2]: <https://github.com/groupmm/libsoni>
[^3]: <https://pypi.org/project/libsoni>

# Acknowledgements
The libsoni package originated from collaboration with various individuals over the past years. We extend our gratitude 
to former and current students, collaborators, and colleagues, including Jonathan Driedger, Angel Villar-Corrales, and 
Tim Zunner, for their support and influence in creating this Python package. This work was funded by the Deutsche 
Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (DFG-MU 2686/15-1) and Grant No. 
328416299 (MU 2686/10-2). The International Audio Laboratories Erlangen are a joint institution of the 
Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.


# References