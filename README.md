# Driver Attack 
### For generating adversarial patches to fool the Driver State Detection Systems


#### `Main.ipynb` contains the main pipeline
#### `advPatch.ipynb` contains the Black Box adversarial attack patch.
#### `Plots` dir will eventually hold the plots generated.
#### `Model_weights` dir will contain the weights of the model. Since size of these weight files are large the weights will be eventually uploaded on drive and download scipt will be added.
#### `data` dir will contain the unzipped version of AUC dataset


# ART
Edited version of `ART` package is added in `toolbox` folder.
+ Install the `ART` package in editor mode using ```pip install -e toolbox/```
+ When initializing the PytorchClassifier instance as done earlier give `ZOO-Adamm` flag as `True`. Currently there is a RuntimeError occuring as shown below:

![](https://github.com/sumanyumuku98/DriverAttack/blob/master/Screenshot_2020-08-27%20Playground%20-%20Jupyter%20Notebook.png)
+ The changes in the `ART` toolbox has been done in `art/estimators/classification/pytorch.py` file. So error has to be resolved in this file only. Satya should look into this:
# TODO
+ Train the model while cross validating and storing the weights
+ Adv Patch
+ Generate Plots and store `.jpeg` in `plots` dir.
