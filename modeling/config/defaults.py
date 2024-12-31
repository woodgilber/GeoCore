from fvcore.common.config import CfgNode

_C = CfgNode()
_C.run_description = ""

#######################################################################################################################
# DATA PARAMETERS #####################################################################################################
# _C is the main configuration object of type CfgNode.
# Contains nested CfgNode objects for organizing parameters related to data, models, optimization, and cross-validation.

_C.sql_parameters = CfgNode()
# list of features to pull from snowflake. Each feature is a class name located in the features folder
_C.sql_parameters.features = ["IngeniousGrid"]
# first table for the left join
_C.sql_parameters.first_table = "IngeniousGrid"
# label class (also located in the features folder)
_C.sql_parameters.labels = ["Ingenious_Wells_TempC_Labels"]
# when training a meta-model, we use labels from other models as features. Warning: you need at least one feature other
# than first_table (that feature will be dropped anyway)
_C.sql_parameters.meta_labels = []
_C.sql_parameters.label_column = "LABEL"
_C.sql_parameters.meta_columns = ["H3_CENTER"]
_C.sql_parameters.test_data = ["IngeniousHotSprings"]
# Observation weights:
_C.sql_parameters.weights_column = None

# some parameters to threshold / filter the data
_C.sql_parameters.params = CfgNode()

# pred labels
_C.sql_parameters.params.positive_well_depth_threshold = 20
_C.sql_parameters.params.negative_well_depth_threshold = 400
_C.sql_parameters.params.pred_score_threshold = 0.9
_C.sql_parameters.params.application_table = "PFA_META_MODEL_3_APPLICATION_SET"

# do not select points that are too close to the test set
_C.sql_parameters.params.test_buffer = 1000

# feature to be turned to a pandas categorical variable
_C.sql_parameters.categorical_features = []


#######################################################################################################################
# MODEL PARAMETERS ####################################################################################################

_C.model = CfgNode()
# model name (= class defined in the models folder)
_C.model.name = "LGBMBinaryClassifier"
# if set to higher than 0, it will drop feature_threshold ratio of the features with the lowest gain before running the
# optimization process. Useful when dealing with a high number of features
_C.model.feature_threshold = 0.0

#######################################################################################################################
# LIGHTGBM PARAMETERS #################################################################################################
_C.model.lightgbm = CfgNode()
_C.model.lightgbm.model_params = CfgNode()
_C.model.lightgbm.model_params.verbose = -1
_C.model.lightgbm.model_params.is_unbalance = True
_C.model.lightgbm.model_params.boosting_type = "gbdt"

_C.model.lightgbm.optim_params = CfgNode()
_C.model.lightgbm.optim_params.verbose = 1
_C.model.lightgbm.optim_params.n_iter = 10
_C.model.lightgbm.optim_params.scoring = "roc_auc"


# search spaces for each parameters, tuple of (min value, max value, scale)
_C.model.lightgbm.optim_params.search_spaces = CfgNode()
_C.model.lightgbm.optim_params.search_spaces.max_bin = (2, 255, "uniform")
_C.model.lightgbm.optim_params.search_spaces.max_depth = (2, 3, "uniform")
_C.model.lightgbm.optim_params.search_spaces.num_leaves = (2, 64, "uniform")
_C.model.lightgbm.optim_params.search_spaces.min_child_samples = (10, 100, "uniform")
_C.model.lightgbm.optim_params.search_spaces.learning_rate = (
    0.001,
    0.1,
    "log-uniform",
)
_C.model.lightgbm.optim_params.search_spaces.min_child_weight = (
    0,
    10,
    "uniform",
)
_C.model.lightgbm.optim_params.search_spaces.n_estimators = (500, 2000, "uniform")
_C.model.lightgbm.optim_params.search_spaces.colsample_bytree = (0.1, 0.5, "uniform")
_C.model.lightgbm.optim_params.search_spaces.subsample = (0.4, 0.8, "uniform")
_C.model.lightgbm.optim_params.search_spaces.subsample_freq = (1, 7, "uniform")
_C.model.lightgbm.optim_params.search_spaces.reg_alpha = (
    0.00000001,
    1.0,
    "log-uniform",
)
_C.model.lightgbm.optim_params.search_spaces.reg_lambda = (
    0.00000001,
    1.0,
    "log-uniform",
)

#######################################################################################################################
# logistic PARAMETERS #################################################################################################
_C.model.logistic = CfgNode()
_C.model.logistic.model_params = CfgNode()
_C.model.logistic.model_params.verbose = 0
_C.model.logistic.model_params.solver = "liblinear"
_C.model.logistic.model_params.max_iter = 1000

_C.model.logistic.optim_params = CfgNode()
_C.model.logistic.optim_params.n_iter = 10
_C.model.logistic.optim_params.search_spaces = CfgNode()
_C.model.logistic.optim_params.search_spaces.penalty = ["l1", "l2"]
_C.model.logistic.optim_params.search_spaces.C = (0.00000001, 1.0, "log-uniform")
_C.model.logistic.optim_params.scoring = "roc_auc"


#######################################################################################################################
# linear svm PARAMETERS ###############################################################################################
_C.model.svm = CfgNode()
_C.model.svm.model_params = CfgNode()
_C.model.svm.model_params.max_iter = 50000
_C.model.svm.model_params.probability = True

_C.model.svm.optim_params = CfgNode()
_C.model.svm.optim_params.n_iter = 10
_C.model.svm.optim_params.search_spaces = CfgNode()
_C.model.svm.optim_params.search_spaces.C = (0.00000001, 1.0, "log-uniform")
_C.model.svm.optim_params.search_spaces.kernel = ["linear", "poly", "rbf"]
_C.model.svm.optim_params.search_spaces.degree = [3, 5, "uniform"]
_C.model.svm.optim_params.search_spaces.gamma = ["scale", "auto"]
_C.model.svm.optim_params.scoring = "roc_auc"

#######################################################################################################################
# linear nn PARAMETERS ################################################################################################
_C.model.nn = CfgNode()
_C.model.nn.model_params = CfgNode()
# number of epochs
_C.model.nn.model_params.max_iter = 200

_C.model.nn.optim_params = CfgNode()
_C.model.nn.optim_params.n_iter = 10
_C.model.nn.optim_params.search_spaces = CfgNode()
_C.model.nn.optim_params.search_spaces.n_layers = (1, 5, "uniform")
# number of neurons is 2**n_neurons
_C.model.nn.optim_params.search_spaces.n_neurons = (4, 8, "uniform")
_C.model.nn.optim_params.search_spaces.activation = ["relu", "tanh"]
_C.model.nn.optim_params.search_spaces.solver = ["adam", "sgd"]
_C.model.nn.optim_params.search_spaces.learning_rate_init = (0.00001, 0.1, "log-uniform")
_C.model.nn.optim_params.search_spaces.learning_rate = ["constant", "invscaling", "adaptive"]
_C.model.nn.optim_params.scoring = "roc_auc"

#######################################################################################################################
# CROSS-VALIDATOR PARAMETERS ##########################################################################################
# cross validator - name of a class in the cross_validators folder
_C.model.cross_validator = "KfoldWBuffer"
_C.model.cross_validator_parameters = CfgNode()
# geospatial splits - number of rows / cols in the grid
_C.model.cross_validator_parameters.nx = 5
_C.model.cross_validator_parameters.ny = 2
_C.model.cross_validator_parameters.n_folds = 10
# buffer distance between splits in km
_C.model.cross_validator_parameters.buffer = 10
# distance in km for agglomerative clustering
_C.model.cross_validator_parameters.clustering_distance = 50
# for LooCV, we we downsample the validation set when evaluating (otherwise we train/evaluate n_samples models) to make
# things faster
_C.model.cross_validator_parameters.downsample_factor = None
