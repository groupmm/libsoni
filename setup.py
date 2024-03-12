from setuptools import setup, find_packages


with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='libsoni',
    version='1.0.1',
    description='A Python Toolbox for Sonifying Music Annotations and Feature Representations',
    author='Yigitcan Özer, Leo Brütting, Simon Schwär, Meinard Müller',
    author_email='yigitcan.oezer@audiolabs-erlangen.de',
    url='https://github.com/groupmm/libsoni',
    download_url='https://github.com/groupmm/libsoni',
    packages=find_packages(exclude=['tests*']),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
    ],
    keywords='',
    license='MIT',
    install_requires=['ipython >= 7.8.0, < 8.0.0',
                      'librosa >= 0.8.0, < 1.0.0',
                      'matplotlib >= 3.1.0, < 4.0.0',
                      'numpy >= 1.17.0, < 2.0.0',
                      'pandas >= 1.0.0, < 2.0.0',
                      'pysoundfile >= 0.9.0, < 1.0.0',
                      'scipy >= 1.7.0, < 2.0.0',
                      'libfmp >= 1.2.0, < 2.0.0'],
    python_requires='>=3.7, <4.0',
    extras_require={
        'tests': ['pytest == 6.2.*'],
        'docs': ['sphinx == 6.2.*',
                 'sphinx_rtd_theme == 1.2.*']
    }
)