## Multi-Task Learning for Emotion Recognition in Conversation

## Getting started

### On Colab

- Upload `src/colab.ipynb` to Colab. 
- Manually upload data folder to Colab's content folder while maintaining the same directory structure. 
- Execute Runtime/Run All 
- Customize training arguments in args dictionary

### Condor or Locally

- Install [conda](https://docs.anaconda.com/anaconda/install/index.html) and [git-lfs](https://git-lfs.github.com/)
- Clone this project and change to the project's root directory
- `git lfs fetch` to fetch compressed model file
- `conda env create -f environment.yaml` to initialize conda environment
- `cd outputs/D2 && tar -xvzf model.tar.gz` to extract compressed model file

#### Condor
- `condor_submit D2.cmd` to run inference and evaluation
- `condor_submit D2_full.cmd` to run training and evaluation

#### Locally
- `python src/pipeline.py` to run inference and evaluation
- `python src/pipeline.py --do_train` to run training and evaluation
- You can find more flags via `python3 src/pipeline.py -h` 
