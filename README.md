# Modulation Classification
This repo contains modules and scripts to train a modulation classifier and an 
implementation for a software defined radio using GNU Radio for real-time 
modulation classification. The module `extract_features.py` contains functions 
to load IQ data, augment it with AWGN, extract features from it, and convert it 
into a format for training a classifier. The module `features.py` contains 
various feature extraction methods implemented from publications in automatic 
modulation classification. The file `train_model.py` trains either a decision 
tree or a feed forward neural network with the specified features and augments
the training data with AWGN with specified SNR levels. The performance of the 
classifier is displayed using the functions in the `plots.py` module. The 
classifier may then be tested on data with different SNR level with the script
`test_model.py`. This uses an implementation of the forward pass of a 
feedforward neural network found in `FeedForwardNetwork.py`. This implementation 
of a neural network is used in the GNU Radio script `modulation_classifier.py` 
which is a real-time simulation of the modulation classifier for a software 
defined radio..

## Dependencies
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn
- GNURadio
