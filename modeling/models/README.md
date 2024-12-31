We define all base classes for our models in the base.py module. The model classes can be pretty much anything you want. Right now, we follow the sklearn convention on both the models and cross validators, but there is no reason why custom formats cannot be implemented in this pipeline.

For each new model we need to register it in the MODEL_REGISTRY

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class SomeModel: 

    def __init__(self,...): 

    def routine1(self, ...):
If you are creating a new file, don't forget to import it into the build.py module since that is the one that is being imported by all the other processes.