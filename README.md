# Machine Learning model for vessel activity

This repo serves as my submission for the Global Fishing Watch project. While I had originally begun with the GeoCore framework, I realized that a large amount of it is hardcoded to specifically work with only the INGENIOUS data set. Instead of parsing through all of the files I decided to start over using tensorflow. I attemped to keep in the spirit of GeoCore though, still using Snowflake to store the data and an ML visualization tool (tensorboard, discussed below) to monitor the experiments. A yaml file is still used to select hyperparameters to test. Because it is intended to work in a similar way, the initial setup is very similar.

## Installation
### Without poetry

1. Install [poetry](https://python-poetry.org/docs/#installation)

2. Create the poetry environment with `poetry install`

## Usage
### Exploratory Data Analysis

The EDA.ipynb walks through my initial exploration of the provided data. This includes verifying the cleanliness of the dataset, inspecting the values, and testing out some transformations for the pipeline.

### Storage
First, create a [free Snowflake account](https://signup.snowflake.com/). While using these scripts there is an assumption that you will have setup your environment variables to enable Snowflake access. I did this with a bash script, but did not commit it this repo as it contains my login credentials. However, it looks generally like this:

```bash
export SNOWFLAKE_DATABASE=XXX # use the name of the database created above
export SNOWFLAKE_SCHEMA=XXX # use the name of the schema created above
export SNOWFLAKE_USER=XXX
export SNOWFLAKE_PASSWORD=XXX
export SNOWFLAKE_ACCOUNT=XXX # find value here https://docs.snowflake.com/en/user-guide/gen-conn-config#using-sql-statements-to-find-your-account-identifier
```

And can be called with `source snowflake_creds.sh`. To upload the data from the trawlers file (store separately) you call the data_pipline script as follows.

```bash
python data_pipeline.py --data pth/to/parquet
```

There is an optional `--shoreline` argument to change the shape file used to calculate the distance to the shoreline if desired.

### Training
Once data has been loaded into the database you can then train your models on it. Multiple models can be created based on the parameters in the `experiment_configs/custom_fishing.yaml` file.

To begin the model training run the following command in your terminal:
```bash
python modeling.py --params experiment_configs/custom_fishing.yaml --table fishing_raw_data
```

The table argument should be changed if the table name was changed in the data_pipeline file. This process will save the separated and scaled training, validation, and test data locally, as well as the models and logs for the Tensorboard tracking.

### Experiment tracking
The experiements are setup to be tracked with Tensorboard. The relevant files are stored locally in the `logs` folder. To view a specific run use the command `tensorboard --logdir 'pth/to/dir'` -- in your terminal. This will promt you to open the following command in your browser:

```
"http://localhost:6006"
```

### Results

A brief summary of the results can be found in the `results.ipynb` notebook. At the beginning I use the model that performed the best for me, but based on randomness and changes to the yaml file you might need to select a different model.

## Citing
`GeoCore` was developed by [Zanskar Geothermal and Minerals](https://www.zanskar.com/).

`Grujic, O., Hossler, T.,  Lipscomb, J., Morrison, R.A., and Smith, C.M. (2025). An efficient and scalable framework to optimize geospatial machine learning. PROCEEDINGS, 50th Workshop on Geothermal Reservoir Engineering Stanford University, Stanford, California, February 10-12, 2025 SGP-TR-229`

Main contributors:

* Thomas Hossler [@defqoon](https://github.com/defqoon)
* Ognjen Grujic [@ogru](https://github.com/ogru)
* Connor Smith [@conster303](https://github.com/conster303)
* Jacob Lipscomb [@jakezanskar](https://github.com/jakezanskar)
* Rachel Morrison  [@rmorrison24](https://github.com/rmorrison24)


## License
The source code of this project is licensed under the [MIT License](https://opensource.org/license/mit). 
