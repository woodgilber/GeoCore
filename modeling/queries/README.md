This is a standalone module that allows for establishing queires to test against the application set. It is seperate from the (test set of)[modeling/datasets/test_data.py] which has h3 cells held out (and buffered) from the application set. 

The data here is called in the (base class)[modeling/models/base.py] BinaryClassifier module - in order to declare a table to join the application set to for testing. 
