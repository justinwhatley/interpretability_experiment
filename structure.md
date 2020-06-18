
Experiment design for model comparisons


├── LICENSE
│
├── README.md                   <- The top-level README for developers using this project.
│
├── setup.py                    <- Make this project pip installable with `pip install -e`
│
├── documentation               <- Relevant program flow charts, reports, etc. Will probably want this to 
│                                  a format where version tracking is possible (e.g., latex, html)
│
├── project                     <- Main project directory
│   │
│   ├── data                    <- Scripts to download or generate data
│   │   ├── external            <- Data from third party sources
│   │   ├── interim             <- Intermediate data that has been transformed
│   │   ├── processed           <- The final, canonical data sets for modeling
│   │   └── raw                 <- The original, immutable data dump (left empty with external data)
│   │
│   ├── experiments             <- For controlled experimentation across modelling strategies 
│   │   ├── run_experiments.py  <- Main experiment script to call set of experiments
│   │   │
│   │   ├── dataset
│   │   │   └── dataloader.py   <- Dataloader designed to provide data to experiement scripts
│   │   │
│   │   ├── figures             <- Saved figures, might make sense to expand this to 'results' more generally
│   │   │
│   │   ├── models              <- Saved models to avoid lengthy retraining when this is not necessary
│   │   │
│   │   ├── experiment_scripts  <- Experiment definition to load appropriate training and test scripts
│   │   │   ├── 01_lr_c_exp.py   
│   │   │   ├── 02_lr_r_exp.py       
│   │   │   └── ...
│   │   │
│   │   ├── testing_scripts     <- Customized testing scripts for experiments
│   │   │   ├── 01_test_class.py   
│   │   │   ├── 02_test_reg.py     
│   │   │   └── ...
│   │   │  
│   │   ├── training_scripts    <- Customized training scripts for experiments
│   │   │   ├── 01_lr_class.py  
│   │   │   ├── 02_lr_reg.py  
│   │   │   ├── 03_lightgbm_class.py  
│   │   │   ├── 04_lightgbm_reg.py  
│   │   │   └── ...
│   │   │ 
│   │   └── config.ini          <- Dataset loading parameters, including column handlings details,
│   │                              path locations (relative and absolute), etc.   
│   │   
│   ├── notebooks               <- For quick experimentation of ideas, new frameworks, etc.
│   │   │                 
│   │   ├── 01_jw_explore.ipynb <- Naming convention is a number (for ordering), creator initials, 
│   │   │                          and a short `_` delimited description
│   │   ├── ...
│   │   │
│   │   ├── figures            
│   │   │
│   │   ├── models              
│   │   │
│   │   └── config.ini            
│   │
│   │
│   └── utils                   <- Common utilities to be used in notebooks and experiment 
│       ├── __init__.py         <- Makes utils a Python module after setup.py is run
│       │
│       ├── models
│       ├── distributed
│       ├── preprocessing
│       ├── setup
│       └── visualization


This is inspired by Cookie Cutter datascience: https://drivendata.github.io/cookiecutter-data-science/ so
verify that initial structure might not better suit your needs