## MlGeoCore 

![plot](/mlgeocore_wf.png)

### What is MlGeoCore? 

This repository is designed to be a light weight core library for common and essential functionality shared in various geospatial machine learning (ML) tasks. 


#### Citing
`MlGeoCore` was developed by Zanskar Geothermal and Minerals. 

We encourage you to cite references to this code as: 

`Smith, C., Hossler, T.,  Lipscomb, J., Morrison, R., and Grujic, O. (2024). An efficient and scalable framework to optimize geospatial machine learning. PROCEEDINGS, 50th Workshop on Geothermal Reservoir Engineering Stanford University, Stanford, California, February 10-12, 2025 SGP-TR-229`

We welcome feedback, bug reports and code contributions from third parties.

### Main contributors

* Thomas Hossler [@defqoon](https://github.com/defqoon)
* Ognjen Grujic [@ogru](https://github.com/ogru)
* Connor Smith [@conster303](https://github.com/conster303)
* Jacob Lipscomb [@jakezanskar](https://github.com/jakezanskar)
* Rachel Morrison  [@rmorrison24](https://github.com/rmorrison24)

### License

The source code of this project is licensed under the [MIT License](LICENSE). Certain libraries in the project dependencies might be distributed under more restrictive open source licenses. TODO: Refine


The structure of each folder builds off of meta's fvcore https://github.com/facebookresearch/fvcore, where each folder contains:

- **Class Files:** Classes are typically located in separate files (one class per file), and can be customized or pull from off the shelf libraries including sklearn, pytorch, etc., (For example)[modeling/models/lgbm_binary_classifier.py]
- **Registry:** Feature that lets us register and access classes by their names.
- **Base File:** Each folder contains a `base.py` file with an abstract base class (e.g., a base model).

This structure allows for easy implementation and flexible tuning of any data type, model architecture, cross validator, and other parts of the library. Modeling in our library is powered by experiment configuration files (.yaml) e.g.,[experiment_configs/first_experiment.yaml](https://github.com/Zanskar-Geothermal/mlgeocore_public/blob/0af1ae46f82aec268396f194675e4ec3bf89da2a/experiment_configs/first_experiment.yaml#L1), which control data and model inputs. 

Any experiments are based on the design of these files. The config system uses Facebook’s [fvcore CfgNode](https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/config.py) class, which itself relies on yet another configuration system [YACS](https://github.com/rbgirshick/yacs).

 Each element of the config file is defined in the default file [config/defaults.py](https://github.com/Zanskar-Geothermal/mlgeocore_public/blob/0af1ae46f82aec268396f194675e4ec3bf89da2a/modeling/config/defaults.py#L1). If a parameter is not defined in the input yaml file, the default value will be used.

## Project Structure

The project structure and the most important files are as follows:

- `dockerfiles`: The directory for dockerfiles to launch a contanerizerd virtual enviroment (CPU or GPU)
- `experiment_configs`: The directory for yaml files to declare experiment desgins
- `modeling/` : The directory for all things modeling.
  - `actions/` : The directory for any actions we perform (e.g., train or apply).
  - `config/` : The directory for the configuration management system for a machine learning or modeling pipeline.
      - `config.py` : Flexible and centralized way to manage configuration parameters. Uses fvcore.common.config.CfgNode to organize and retrieve parameters hierarchically.
      - `defaults.py`: Contains nested CfgNode objects for organizing parameters related to data, models, optimization, and cross-validation.
  - `cross_validators/` : The directory for custom cross validation classes used and called for splitting traing/validation data. 
    - `base.py/` :  abstract base classes that outline the structure and functionality that other classes must implement.
    - `build.py/` :  factory or registry for dynamically creating instances of classes based on configuration or user input.
  - `dataset/` : The directory for custom datasets classes used and called for splitting traing/validation data. 
      - `base.py/` :  abstract base classes that outline the structure and functionality that other classes must implement.
    - `build.py/` :  factory or registry for dynamically creating instances of classes based on configuration or user input.
  - `models/` : The directory for any ml models (supervised or unsupervised). 
    - `base.py/` :  abstract base classes that outline the structure and functionality that other classes must implement.
    - `build.py/` :  factory or registry for dynamically creating instances of classes based on configuration or user input.
  - `queries/` : The directory for custom queires to log performance vs the application set
  - `utils/` : The directory for custom utility modules and connectors that are called by classes in modeling, datasets, cross validaiton, etc., 
- `poetry.lock`: resolves all dependencies and their sub-dependencies in the pyproject.toml file
- `pyproject.toml`: Contain the python requirements and provides specific for a python environment.
- `README.md` : The README file for the project.


For further details on the contents see the readme's within each folder.

## Installation: 

### With docker (recommended)

Build the docker image 
```
cd modeling
# cpu version
docker build -f dockerfiles/cpu/Dockerfile -t modeling-image .

# gpu version (beta)
docker build -f dockerfiles/gpu/Dockerfile.gpu -t modeling-image .
```

Create the docker container and run the code from within. Using the GPU version is recommended
when dealing with large datasets. 

See .toml file ```tool.poetry.scripts``` to fire up command line decorator actions.

### Without docker

Step 1: Install [poetry](https://python-poetry.org/) with `python3 -m pip install poetry`

Step 2: Set the environmental variable for poetry to use local virtual environment: `export POETRY_VIRTUALENVS_IN_PROJECT=true`

Step 3: Create the virtual environment by typing `poetry install`

Step 4: Open poetry shell with: poetry shell

Step 5: See .toml file ```tool.poetry.scripts``` to fire up actions. 


## Modeling

Most of modeling will be done with simple manipulations of the config files.  The structure of the config files is as follows:

```yaml 
run_description: Test run on INGENIOUS data and aoi 

sql_parameters:
  features: 
    - Ingenious_Outline_H3Grid
    - Ingenious_Depth_to_Basement
    - Ingenious_CBA_HGM_Gravity
    - Ingenious_Earthquake_Density_n100a15
    - Ingenious_MT_Conductance_Layers
    - Ingenious_Geodetic_Layers
    - Ingenious_Heatflow
    - Ingenious_RTP_HGM_Magnetic
    - Ingenious_Quaternary_Volcanics_Distance
    - Ingenious_Fault_Layers

  first_table: Ingenious_Outline_H3Grid
  labels: 
    - Ingenious_Wells_TempC_Labels
  label_column: LABEL
  meta_columns:
    - H3_PARENT
    - H3_CENTER
    - LONGITUDE
    - LATITUDE
  test_data: 
    - IngeniousHotSprings
  params:
    well_depth_threshold: 100
model: 
  name: LGBMBinaryClassifier
  lightgbm:
    model_params:
      verbose : -1
      boosting_type: 'gbdt'
    optim_params:
      verbose: 0
      n_iter : 30
      scoring: "roc_auc"
      percentiles : [0.05,0.1,0.5,0.9,0.95]
      search_spaces:
        max_depth : [2, 8, 'uniform']
        num_leaves : [2, 512, 'uniform']
        min_child_samples : [15, 100, 'uniform']
        learning_rate : [0.001, 0.2,'log-uniform']
        n_estimators : [100, 120, 'uniform']
        colsample_bytree : [0.2, 0.6, 'uniform']
        subsample :  [0.2, 0.6, 'uniform']
        subsample_freq :  [1, 7, 'uniform']
        reg_alpha :  [0.00000001, 1.0, 'log-uniform']
        reg_lambda :  [0.00000001, 1.0, 'log-uniform']
  cross_validator: BlockCV
  cross_validator_parameters: 
    nx : 5
    ny : 2
    n_folds: 5
    buffer : 10
    clustering_distance: 10
```

# Scripts and command line decorators

### Training

The training script is central to the modeling pipeline and works by manipulating configuration files (.yaml). These files define data sources, model parameters, and other experimental settings. By specifying the configuration file and experiment name, you can train a model, optimize its parameters, and log artifacts for analysis.

#### To train a model:

`train -c <config_file.yaml> -e <experiment_name_here>`

#### Flags too add in command decorator:

Below are flag requiremnts/options to add to the training module (train.py))[modeling/actions/train.py]. These include:

- `-c, --config_path`  
  Path to the configuration file. This is required.

- `-e, --experiment_name`  
  Name of the MLflow experiment. This is required.

- `-h, --cache`  
  Optional flag to enable cache mode. If `True`, uses locally cached data instead of pulling from Snowflake. Default is `False`.

- `-s, --save_test`  
  Optional flag to save the test set locally. Default is `True`.

- `-n, --run_name`  
  Optional custom run name for the MLflow experiment. If not provided, a random run name is assigned.

- `-d, --debug`  
  Optional flag to enable debug mode. Useful for troubleshooting. Default is `False`.

- `-p, --profile`  
  Optional flag to enable profiling for performance analysis. Default is `False`.

- `l, --profile`  
  Optional flag to skip Snowflake operations and only run locally - requires -h to be true. Default is `False`.

For instance: 

```poetry run train -c ./experiment_configs/lightgbm_gbdt_model_ingenious.yaml -e MLGEO_CORE -h True -l True``` 

- will run a model that has an experiment named MLGEO_CORE
- will called cached data (`-h`)
- and will not perform any snowflake opeartions (`-l`)

#### Key Features:

1. **Data Handling**  
   - If a test, train, and application set is pre-established we can utilize cacheing to quickly run new experiments.
   - If we have a snowflake database we can connect and pull data for splitting data it into training/testing and application sets.  
   
2. **Model Optimization**  
   - Automatically tunes hyperparameters and fits the best-performing model.  

3. **Logging Artifacts**  
   - Logs feature importance plots, PR curves, ROC curves, and other metrics using MLflow.  
   - Saves models and parameters for reproducibility.  


#### Example Workflow:

Run a training job:  
`train -c config.yaml -e PFA`
#### Outputs:

- **/runs/ outputs**: Test train and application result `.pq` files (features, labels, predictions, model) cross validation `.csv` file, map outputs `.html`
- **Snowflake Table**: If we use snowflake - we save application results in a table named `<datbase>.<experiment_name>.<run_name>_APPLICATION_SET`.  
- **MLflow Dashboard**: Visualize experiments by running:  
  `poetry run mlflow ui`

#### Training Script Workflow:

1. Pulls data from Snowflake and splits it into train and application sets.  
2. Optimizes model parameters using the config file and fits the final model.  
3. Logs artifacts like importance plots, PR, and ROC plots.  
4. Saves the model and metadata.  
5. Applies the model on the application set and pushes results to Snowflake.

---

## Experiment Tracking


We perform experiment tracking using MLFlow. Each developer has his own local MLFlow instance. 

In order to view your experiment, start the mlflow server using `poetry run mlflow-ui` in your terminal.

This decorator is declared in the toml as

[`mlflow-ui = "modeling.actions.mlflow_ui:open_mlflow_ui"`]

and will create a UI page hosted at
```
"http://localhost:5050"
```
Extensive documentation about MLFlow can be found [here](https://mlflow.org/docs/latest/index.html).

- MLflow tracks all models under the provided experiment name.  
- The run name (e.g., `mad-dog-124`) corresponds to the Snowflake table storing application results.  


## Snowflake connection 

If you have a snowflake account to connect to, export your snowflake user name and password with
- `export SNOWFLAKE_USER=...` 
- `export SNOWFLAKE_PASSWORD=...` 
- `export SNOWFLAKE_ACCOUNT=...` 
- `export SNOWFLAKE_ROLE=...` 
- `export SNOWFLAKE_DATABASE=...`
- `export SNOWFLAKE_SCHEMA=...` 
- `export SNOWFLAKE_WAREHOUSE=...` 


For info on customizing the snowflake connection see [modeling/utils/snowflake.py](https://github.com/Zanskar-Geothermal/mlgeocore_public/blob/855f50252e896e2b6658936cb6b717078bd144ed/modeling/utils/snowflake.py#L1)


## Pre-Commit
We use pre-commit hooks to format our code in a unified way.

Pre-commit is installed within the poetry structure. With pre-commit it will trigger 'hooks' upon each `git commit -m ...` command. The hooks will be applied on all the files in the commit. A hook is nothing but a script specified in `.pre-commit-config.yaml`.

More information about pre-commit can be found [here](https://pre-commit.com).

