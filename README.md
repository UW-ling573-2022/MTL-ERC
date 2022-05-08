## Multi-Task Learning for Emotion Recognition in Conversation

## Getting started

### On Colab

- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UW-ling573-2022/teamCDFJ/blob/main/src/colab.ipynb) (Make sure you allow Google Colab to [access this private repo](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=Rmai0dD30XzL))
- Execute Runtime/Run All 
- Customize training arguments in `kwargs` dictionary
  - Change `do_train` to `False` to only run inference and evaluation.

### Condor or Locally

- Install [conda](https://docs.anaconda.com/anaconda/install/index.html)
- Clone this project and change to the project's root directory
- `conda env create -f environment.yaml` to initialize conda environment
- Put the downloaded the `pytorch_model.bin`model file from [Google Drive](https://drive.google.com/file/d/11gUWjXurmcthj6UGVoaOk-N2dE_Q1w3N/view?usp=sharing)
in `outputs/D3`

#### Condor
- `condor_submit D3.cmd` to run inference and evaluation
- `condor_submit D3_full.cmd` to run training and evaluation

#### Locally
- `python src/pipeline.py` to run inference and evaluation
- `python src/pipeline.py --do_train` to run training and evaluation
- You can find more flags via `python3 src/pipeline.py -h` 
