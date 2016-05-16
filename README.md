# lrh
Learning RNN Hierarchies

`about.md` has the basic details of what the code is trying to do and why.

* Layers are defined in `layers/` along with tests
* All layers derive from abstract base class defined in `base.py`
* All data and data prepping scripts are in `data/` folder
* `network.py` has tools for taking in a list as a model and doing layer by layer forward/backward pass, getting gradients, setting/getting parameters 
* `train_ptb.py` trains a model on Penn Tree Bank text file, which has to be placed in the `data/` folder
* `train_mnist.py` trains a model on Sequential MNIST. `mnist.pkl.gz` has to be placed in `data/` folder
* As the network trains, logs are generated. Final logs and models are stored as pickle objects in `results/experiment_name`, where `experiment_name` is a string defined in `train_` scripts

##Requires:
* Numpy
* Scipy (for one special function to calculate entropy)
* matplotlib
* climin

