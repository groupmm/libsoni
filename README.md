[![Python package using Conda](https://github.com/groupmm/libsoni/actions/workflows/test_conda.yml/badge.svg)](https://github.com/groupmm/libsoni/actions/workflows/test_conda.yml)
[![Python package using pip](https://github.com/groupmm/libsoni/actions/workflows/test_pip.yml/badge.svg)](https://github.com/groupmm/libsoni/actions/workflows/test_pip.yml)


<table border="0">
  <tr>
    <td><img src=https://github.com/groupmm/libsoni/blob/main/docs/build/html/_static/libsoni_logo.png alt="libsoni logo" width="1000"></td>
    <td><h2>libsoni: A Python Toolbox for Sonifying Music Annotations and Feature Representations</h2>
<br> <br>
</td>
  </tr>
</table>

``libsoni`` is an open-source Python toolbox tailored for the sonification of music annotations and feature representations. 
By employing explicit and easy-to-understand sound synthesis techniques, the toolbox offers functionalities
for generating and triggering sound events, enabling the sonification of spectral, harmonic, tonal, melodic,
and rhythmic aspects. Unlike existing software libraries focused on creative applications of sound generation, 
the toolbox is designed to meet the specific needs of MIR researchers and educators. It aims to simplify the process
of music exploration, promoting a more intuitive and efficient approach to data analysis by enabling users to interact 
with their data in acoustically meaningful ways. 

See the [API documentation](https://groupmm.github.io/libsoni/build/html/index.html) for a detailed view of the provided functions in ``libsoni``.

## Installation Guide
We outline two primary methods for setting up `libsoni` using pip and setting up a dedicated environment.

### Method I: Installing with pip
Utilize Python's package manager, pip, for a straightforward installation of `libsoni`:

```
pip install libsoni
```
Note: We advise performing this installation within a Python environment (such as conda or a virtual environment) 
to prevent any conflicts with other packages. Ensure your environment runs Python 3.7 or higher.

### Method II: Setting Up a Conda Environment
Alternatively, you can create a conda environment specifically for `libsoni` by downloading this repository and installing the library in development mode. This approach not only installs `libsoni` but also includes necessary packages for running the demo notebooks with `jupyter`. One way to achieve this would be using the following commands:

```
conda create -n libsoni python=3.11 flit
conda activate libsoni
flit install --symlink --deps all
```


## Running Example Notebooks
To explore ``libsoni`` through example notebooks:

1. **Install ``libsoni``:** Prior to cloning the repository and running the notebooks, ensure libsoni and its dependencies are installed (as described above).
2. **Clone the repository:** Download the ``libsoni`` repository to your local machine using the following git command:
   
```
git clone https://github.com/groupmm/libsoni.git
```

3. **Install Jupyter:** If not already installed via the conda environment setup, install Jupyter to run the notebooks:

```
pip install jupyter
```

4. **Launch Jupyter Notebook:** Start the Jupyter notebook server by executing: 
```
jupyter notebook
```
This will open a browser window from where you can navigate to and open the example notebooks.

## Contributing

We are happy for suggestions and contributions.  We would be grateful for either
directly contacting us via email (meinard.mueller@audiolabs-erlangen.de) or for creating 
an issue in our GitHub repository. Please do not submit a pull request without prior consultation
with us.

## License
The code for this toolbox is published under an [MIT license](LICENSE).
This does not apply to the data files:
* Schubert songs are taken from the [Schubert Winterreise Dataset](https://zenodo.org/record/4122060). 
* Recording of the cantata *Ach Gott und Herr* by Bach is taken fom [Bach10 Dataset](https://labsites.rochester.edu/air/datasets/Bach10%20Dataset_v1.0.pdf).
* Recording of *Locus Iste* by Anton Bruckner is taken from the [Dagstuhl Choir Set](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiJ1JnT9uuEAxXclP0HHUOXC4EQFnoECBMQAQ&url=https%3A%2F%2Fwww.audiolabs-erlangen.de%2Fresources%2FMIR%2F2020-DagstuhlChoirSet&usg=AOvVaw1sPox9R_Sh1eh5hqV2kgNs&opi=89978449).
* Custom piano audio samples are taken from the [Single Note Database (SNDB)](https://github.com/audiolabs/SNDB).
* Other audio files are taken from the [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html).

## References

Yigitcan Özer, Leo Brütting, Simon Schwär, and Meinard Müller. [libsoni: A Python Toolbox for Sonifying Music Annotations and Feature Representations](https://joss.theoj.org/papers/10.21105/joss.06524). Journal of Open Source Software (JOSS), 9(96): 1–6, 2024.

Meinard Müller and Frank Zalkow. [libfmp: A Python Package for Fundamentals of Music Processing](https://joss.theoj.org/papers/10.21105/joss.03326). Journal of Open Source Software (JOSS), 6(63), 2021.

## Acknowledgements

The libsoni package originated from collaboration with various individuals over the past
years. We extend our gratitude to former and current students, collaborators, 
and colleagues, including Jonathan Driedger, Angel Villar-Corrales, and Tim Zunner, 
for their support and influence in creating this Python package. This work was 
funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)
under Grant No. 500643750 ([DFG-MU 2686/15-1](https://audiolabs-erlangen.de/fau/professor/mueller/projects/learn)) and 
Grant No. 328416299 ([MU 2686/10-2](https://audiolabs-erlangen.de/fau/professor/mueller/projects/sereco2)).
The [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/) are a joint institution of the 
[Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for 
Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).

