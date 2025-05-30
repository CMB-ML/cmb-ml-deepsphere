# Adams DeepSphere implementation within CMB-ML

## Getting Started

cmb-ml should be downloaded as a package with the following steps:

- first clone cmb-ml using `git clone https://github.com/CMB-ML/cmb-ml.git`
- cd into cmb-ml: `cd cmb-ml`
- switch to the `dev-rm-finish` with `git checkout dev-rm-finish`
- install the requirements with conda: `conda env create -f env.yaml`
- activate the environment: `conda activate cmb-ml`
- ensure pip is within this environment: `which pip`
- install package: `pip install .` or `pip install -e .` to allow for editing

Now this repository can be installed:

- cd out of cmb-ml: `cd ..`
- clone this repo: `git clone https://github.com/CMB-ML/cmb-ml-deepsphere.git`
- install the requirements: `conda env update -n cmb-ml -f env.yaml`
- set the CMB_ML_DATA environment variable: `export CMB_ML_DATA=/path/to/cmb_data`
- main file should be executable now: `python main_deepsphere.py`

