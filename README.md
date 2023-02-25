Applying BERT in Protein Classification
![image](https://user-images.githubusercontent.com/34483849/221366597-066d0952-124f-4658-afbf-ea026806084d.png)
==============================

- This is a couse project (Applied Deep Learning) provided by LMU. The project is supervised by Emilio Dorigatti.

- The protein sequence order determines their properties. Therefore, we can use the sequantial information to classify the proteins. If we treat each component as the letter/word in text, the protein classification is similar to text classification problem. 

- In this project, we applied a well-know NLP model, BERT (https://arxiv.org/abs/1810.04805), to dig information from protein sequences and predict their types. We mask the protein sequence the same way as in BERT, and we have two goals in model training:
    1. Reduce the error in predicting the masked taken;
    2. 
- We use 15% of data for testing, in the rest 85%, we use 80% for model training, 20% for model validation.

- How to run the code:
    1. find and open the script: src\train.py (https://github.com/ZhiweiCheng2020/Applied_DL_Bert/blob/main/src/train.py);
    2. set parameter *len_all*: the number of seqs you would like to include in model training;
    3. set parameter *lr*: learning rate;
    4. set parameter *num_epochs* and *batch_size*: number of epochs and batch size;
    5. set parameter *ebd_dim*: number of Encoder Layers;
    6. set parameter *num_head*: number of heads in Multi-Head Attention;
    7. You are off to go! run *train.py* under the project root path, you will find out the model perfermance under folder */results*, and the trained model is saved under folder */models*.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
