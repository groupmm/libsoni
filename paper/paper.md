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
signals into feature representations such as spectrograms, chromagrams, or tempograms~\citep{Mueller15_FMP_SPRINGER}.
Furthermore, musically relevant annotations such as beats, chords, keys, or structure boundaries become
indispensable for training and evaluating MIR approaches.

When evaluating and enhancing MIR systems, it is crucial to thoroughly examine the properties of feature representations
and annotations to gain a deeper understanding of algorithmic behavior and the underlying data. In the musical context,
alongside conventional data visualization techniques, data sonification techniques are emerging as a promising avenue 
for providing auditory feedback on extracted features or annotated information. This is particularly advantageous given 
the finely tuned human perception to subtle variations in frequency and timing within the musical domain.

This paper introduces \emph{libsoni}, an open-source Python toolbox tailored for the sonification of music annotations 
and feature representations. By employing explicit and easy-to-understand sound synthesis techniques, libsoni offers 
functionalities for generating and triggering sound events, enabling the sonification of spectral, harmonic, tonal, 
melodic, and rhythmic aspects. Unlike existing software libraries focused on  creative applications of sound generation,
libsoni is designed to meet the specific needs of MIR researchers and educators. It aims to simplify the process of 
music exploration, promoting a more intuitive and efficient approach to data analysis by enabling users to interact with 
their data in acoustically meaningful ways. As a result, libsoni not only improves the analytical capabilities of music
scientists but also opens up new avenues for innovative music analysis and discovery.
%
Furthermore, libsoni provides well-documented and stand-alone functions covering all essential building blocks crucial 
for both sound generation and sonification, enabling users to efficiently apply and easily extend the methods. 
Additionally, the toolbox includes educational Jupyter notebooks with illustrative code examples demonstrating the 
application of sonification and visualization methods to deepen understanding within specific MIR scenarios.

# Acknowledgements

# References
