# Config file

In GeoCore, we adopt a single config file to control the experiments (data, model, CV etc.). 
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