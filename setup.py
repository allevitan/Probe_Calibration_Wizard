import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="probe_calibration_wizard",
    version="0.1.0",
    python_requires='>3.7', # recommended minimum version for pytorch
    author="Abe Levitan",
    author_email="alevitan@mit.edu",
    description="Graphical tool for manipulating probe calibrations for RPI and ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allevitan/Probe_Calibration_Wizard",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", # 2.0 has better colormaps which are used by default
        "PyQt5",
        "pyzmq",
        "torch>=1.9.0", #1.9.0 supports autograd on indexed complex tensors
        "cdtools>=0.2.0", # This is hosted on a private repo because it allows for *gasp* ptychographic reconstructions. Contact Abe for access.
    ],
    entry_points={
        'console_scripts': [
            'pcw=probe_calibration_wizard.__main__:main'
        ]
    },

    package_dir={"": "src"},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

