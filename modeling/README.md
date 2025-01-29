## Philosophy

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