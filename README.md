## Multi-Task Learning for Emotion Recognition in Conversation

## Getting started

### On Colab

- Open `src/colab.ipynb` in Colab.  
- Execute Runtime/Run All 
- Customize training arguments in args dictionary

### Condor or Locally

- Install [conda](https://docs.anaconda.com/anaconda/install/index.html)
- Clone this project and change to the project's root directory
- `conda env create -f environment.yaml` to initialize conda environment
- `cd outputs/D2`, download the model file from [Google Drive](https://drive.google.com/file/d/11gUWjXurmcthj6UGVoaOk-N2dE_Q1w3N/view?usp=sharing) via UW Google Account and extract compressed model file

#### Condor
- `condor_submit D2.cmd` to run inference and evaluation
- `condor_submit D2_full.cmd` to run training and evaluation

#### Locally
- `python src/pipeline.py` to run inference and evaluation
- `python src/pipeline.py --do_train` to run training and evaluation
- You can find more flags via `python3 src/pipeline.py -h` 
