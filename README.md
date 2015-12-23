# nn-learn
Modded code from I Am Trask to learn neural nets with additional demo code to visualize in real time

##Setup
Assumes you have python 2.7 with pip and virtualenv. 
Learn more about installing these tools for Mac [here](https://hackercodex.com/guide/python-development-environment-on-mac-osx/).

On Ubuntu linux:
```
sudo apt-get install build-essential git
sudo apt-get install python-dev python-virtualenv
```

Clone [this repo](https://github.com/Archethought/nn-learn). Then setup a virtual environment, source it and install the dependencies:

```
cd nn-learn
virtualenv venv
pip install -r requirements
```

N.B. this requirements.txt is for a small python data science sandbox, it has other tools you might not need. Delete what you don't want, some of this stuff takes time to install. You will need matplotlib for sure.

##Execution
Start two terminals

###Terminal 1
```
. ./venv/bin/activate
touch l2_error.csv
python viz.py
```

The plot window should appear. You can leave Terminal 1 running, the plot will refresh each time the file l2_error.csv is updated.


###Terminal 2
```
. ./venv/bin/activate
python nn.py > l2_error.csv
```

You can kill the nn.py run at any time, especially as the layer 2 error grows small enough.

##TODO

* Multi-plot using activations and weights to see how they change over time with respect to the l2_error.
* Use the [Part 2](https://iamtrask.github.io//2015/07/27/python-network-part2/) code with optimizations for alpha learning rate and gradient descent to compare against unoptimized code.


