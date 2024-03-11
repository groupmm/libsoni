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

# Design Choices
When designing the Python toolbox libsoni, we had several objectives in mind. Firstly, we aimed to maintain close 
connections with existing sonification methods provided in in librosa[@McFeeRLEMBN15_librosa_Python] and 
libfmp[@MuellerZ21_libfmp_JOSS]. Secondly, we re-implemented and included all necessary components 
(e.g., sound generators based on sinusoidal models and click sounds), even though similar basic functionality is 
available in other Python packages such as librosa and libfmp. By doing so, libsoni offers a coherent API along with 
convenient but easily modifiable parameter presets. Additionally, the implementations are more efficient than previous 
software. Thirdly, we adopted many design principles suggested by librosa[@McFeeRLEMBN15_librosa_Python]  
and detailed in [@McFeeKCSBB19_OpenSourcePractices_IEEE-SPM] to lower the entry barrier for students and 
researchers who may not be coding experts. This includes maintaining an explicit and straightforward programming 
style with a flat, functional hierarchy to facilitate ease of use and comprehension. The source code for
libsoni, along with comprehensive API documentation, is publicly accessible through a dedicated GitHub
repository [^3]. We showcase all components, including introductions to MIR scenarios, illustrations, and sound examples
via Jupyter notebooks.  Finally, we have included the toolbox in the Python Package Index (PyPI), enabling
installation with the standard Python package manager, pip [^4].

[^3]: <https://github.com/groupmm/libsoni>
[^4]: <https://groupmm.github.io/libsoni>

# Acknowledgements
The libsoni package originated from collaboration with various individuals over the past years. We extend our gratitude 
to former and current students, collaborators, and colleagues, including Jonathan Driedger, Angel Villar-Corrales, and 
Tim Zunner, for their support and influence in creating this Python package. This work was funded by the Deutsche 
Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (DFG-MU 2686/15-1) and Grant No. 
328416299 (MU 2686/10-2). The International Audio Laboratories Erlangen are a joint institution of the 
Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.


# References