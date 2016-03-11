## Neural Networks
processing data by training neural nets

### Contents

* neuralnet.c contains a CPU implementation of a neural net that runs on the MNIST handwritten numeral dataset found here http://yann.lecun.com/exdb/mnist/
* NN_.cu contains a GPU implementation of a neural network that runs on the MNIST handwritten numeral fdataset found here http://yann.lecun.com/exdb/mnist/
* RNN.cu contains a GPU implementation of a Recurrent Neural Net that runs on the GZTAN music dataset for genre classification, found here http://marsyas.info/downloads/datasets.html

### Requirements

for neuralnet.c
* gcc

for NN_.cu
* nvidia cuda toolkit 7.5
* gcc

for RNN.cu
* nvidia cuda toolkit 7.5
* gcc
* libfftw3
* libsndfile


### Build
to build all three
```
make
```

to build just neuralnet.c
```
make run
```

to build just NN_.cu
```
make srun
```

to build just RNN.cu
```
make Rnn
```
note: the RNN code is only set up to compile for cuda architecture 5.2 (Maxwell). To compile for Kepler gpus like the Tesla series, change the '52's near the top of the makefile to '30' or '32' or whichever is appropriate for your particular card.  

### Run
to run neuralnet.c
```
./run trainingimages traininglabels testimages testlabels num_iterations alpha
```

to run NN_.cu
```
./run trainingimages traininglabels testimages testlabels num_iterations alpha
```

to run RNN.cu
```
./Rnn classical <all the classical tracks> jazz <all the jazz tracks> metal <all the metal tracks> pop <all the pop tracks> num_iterations alpha
```
