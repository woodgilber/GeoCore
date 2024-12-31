This is a standalone module that enables easy creation and addition of new datasets into the modeling pipeline.

Basic usage
Note: All the code below can be run inside a Jupyter notebook.

We use the function build_dataset from the build.py module. The function takes configuration file as an input, along with the parameters dictionary and the snowflake connection. Very basic usage:
```
from modeling.datasets.features import *
from modeling.datasets.build import FEATURES_REGISTRY, build_dataset

import snowflake.connector
import os

cnx = snowflake.connector.connect(
            user      = os.getenv("SNOWFLAKE_USER"),
            password  = os.getenv("SNOWFLAKE_PASSWORD"),
            account   = os.getenv("SNOWFLAKE_ACCOUNT"),
            role      = os.getenv("SNOWFLAKE_ROLE"),
            warehouse = os.getenv("WAREHOUSE"),
            database  = os.getenv("SNOWFLAKE_DATABASE"),
            schema    = os.getenv("SNOWFLAKE_SCHEMA"),
        )


# To see all declared feature sets type: 
FEATURES_REGISTRY.__contains__

# Generate a training/application set with a certain number of features: 
cfg = {
    'features' : 
    [
       "IngeniousGrid"
     , "IngeniousConductanceLowerCrust"
     , "IngeniousGeodeticShear"
    ],
    "first_table" : "IngeniousGrid",
    "labels" : "IngeniousLabels",
    "label_column" : "LABEL",
    "meta_columns" : ["H3_CENTER","LONGITUDE", "LATITUDE", "PARENT"]
}
params = {

}

X, Y, meta = build_dataset(cfg, params, cnx) 
```
Here, cfg is a configuration dictionary that declares the feature sets we want to use. Note that these are declared as classes in their respective files within this package.

features is a list specifying all the feature sets we are using
first_table this is the declaration of the first table we plan on using in join. Right now, we are doing left joins, hence this would likely be the grid.
labels feature set of the labels
label_column the labels feature set table should have this column that will be isolated for ML model training.
meta_columns a list of meta columns contained in the final table join. The columns in this list can be used in spatial cross validation or for other purposes.
The params dictionary contains the parameters we are varying in our ML modeling workflow. These parameters are passed to the run_sql routine of each feature set (see below) and the associated sql_code is expected to be parameterized as a formated string.

The return is a tuple of X (features), Y(labels), and meta (the meta columns). All three are data frames with the index corresponding to the column that was used to join all the feature sets together (likely the H3 index).

Declaring a new feature set
The first component are the feature sets that inherit from the BaseFeatures class (in base.py). This class contains routines that call Snowflake to pull the features, perform checks on the tables, and provide basic get functionalities. To define a new feature set we simply do the following:
```
from .base import BaseFeatures
from .build import FEATURES_REGISTRY

import logging

logger = logging.getLogger(__name__)
# To add a new feature set, simply use the template below. 
# All you need to do is provide the name of the feature set in the class definition
# and fill out the required and optional fields. 

@FEATURES_REGISTRY.register()
class SomeDataOrFeatureSet(BaseFeatures): 

    database = 'DATABASE or None'
    schema = 'SCHEMA or None'
    table_name = 'INGENIOUS_GRID' # Required
    index_column = 'H3_BLOCKS' # Required

    sql_code = ''' 
        with grid as (
            select * from some table
        )
        , feature_set_1 as (
            select 
            a.*, 
            b.* 
            from grid
            left join some_features
            on a.h3_index = b.h3_index
            where a.value > {threshold_value} -- can be fixed too
        )
        select * from feature_set_1
    ''' 
```
Key components are: @FEATURES_REGISTRY.register() This is a required component since it allows the rest of the code to reference this class by name. The rest of the data in the feature set definition are all class properties. Property:

database specifies the database where the target feature table is located in Snowflake
schema specifies the schema where the target table is located in Snowflake
table_name specifies the name of the table in Snowflake
index_column specifies the index column we plan on using for joining with other data. This is expected to be the H3 index.
sql_code This is the string that specifies the feature table generation code.