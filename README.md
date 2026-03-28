# Repository for *MagNet: A Bounded Mahalanobis Architecture for Robust and Interpretable Pan-Cancer Diagnostics*
![A horizontal flow chart depicting a three-phase system pipeline. Phase 1, "Data Curation," shows raw micro-RNA data passing through feature extraction modules to select biomarkers. An arrow leads to Phase 2, "Distance-based Classification," containing a Regularized MLP Backbone that feeds into a Diagonal Mahalanobis layer, represented visually by a scatter plot of clustered, multi-colored data points. An arrow leads to Phase 3, "Uncertainty and Safety," where a Monte Carlo Sampling module with a loop indicating repeated passes outputs either a Diagnostic Prediction, represented by biological cell icons, or an Out-Of-Distribution Safety Rejection.](MagNetOverview.png)

## Setup

This project requires **Python 3.10+**. We recommend using a virtual environment (like [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)) to manage dependencies.


### Set up the environment
Create and activate a conda environment:
```
micromamba create -f environment.yml
micromamba activate magnet
```

### Hardware Acceleration (GPU Support)
This codebase is optimized to scale across multiple CPUs and GPUs using **Ray Tune**. 

If you are running this on a machine with NVIDIA GPUs, ensure you have the appropriate CUDA drivers installed so that TensorFlow, PyTorch, and XGBoost can utilize hardware acceleration. Ray will automatically detect available GPUs and scale the concurrent hyperparameter tuning trials accordingly.


## Prepare the Data
Download the file [GSE211692_RAW.tar](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211692) and place it into the data folder.

(After this step you can alternatively run the SLURM script ```submit.sub```to run the full code.)
### Preprocess the Data
```
bash preprocess.sh 'data/GSE211692_RAW.tar' 'input' 'data/mapping_file.txt' 
```
(this may take a while) \
**NOTE:** This will create a working directory named 'input' and delete it after the data preprocessing has finished - if you already have a folder with that title in the directory, it will be corrupted.

## Rund the Model
To train MagNet on Task 1 (Tissue-of-Origin) run
```
python3 -u train.py --comb 2 --baseline OURS
```
To train MagNet on Task 2 (Disease) run
```
python3 -u train.py --comb 0 --baseline OURS
```
Exchange the ```--baseline``` parameter for TABNET, HEAD, 1D, XGB, RF, or LR to run the other models.

## Evaluation
Run
```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.allow_errors=True evaluation.ipynb 
```
Note that some of the evaluation plots require to run at least both tasks of MagNet.
