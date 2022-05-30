## Multi-Task Learning for Emotion Recognition in Conversation

## Getting started

### On Colab

- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UW-ling573-2022/teamCDFJ/blob/main/src/colab.ipynb) (Make sure you allow Google Colab to [access this private repo](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=Rmai0dD30XzL))
- Execute Runtime/Run All 
- Customize training arguments in `kwargs` dictionary
  - Change `do_train` to `False` to only run inference and evaluation.

### Condor or Locally

- Install [conda](https://docs.anaconda.com/anaconda/install/index.html)
- Clone this project recursively (since data folder is a submodule) and change to the project's root directory
- `conda env create -f environment.yaml` to initialize conda environment
- Put the downloaded the `pytorch_model.bin` model file from [Google Drive](https://drive.google.com/uc?id=1kZ8RmDj8K3HihmUiW2gJu8iyZ82cpbex) as `pytorch_model.bin`
in `outputs/D4`

#### Condor
- `condor_submit D4.cmd` to run inference and evaluation
- `condor_submit D4_full.cmd` to run training and evaluation

#### Locally
- `python src/pipeline.py` to run inference and evaluation
- `python src/pipeline.py --do_train` to run training and evaluation
- You can find more flags via `python3 src/pipeline.py -h` 
