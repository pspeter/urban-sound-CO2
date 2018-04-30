# urban-sound-CO2
Urban Sound Challenge project as part of our university course "Computational Intelligence 2".

## Setup

After cloning or pulling the repository, use your shell to navigate to the project folder and 
use the following command to create a new virtual environment:

```python -m venv venv\urbansound```

Afterwards, activate the environment using:

*Windows*: ```venv\urbansound\Scripts\activate.bat```  
*Mac*: ```source activate venv\urbansound\bin\activate```

Then, install the required packages using:

```pip install -r requirements.txt```

Now you're all set!

## Adding a new package

You can use pip to install a new package:

```pip install <package-name>```

When doing so, always remember to update the requirements.txt by using the command

```pip freeze > requirements.txt```

## Reference

- [URBAN SOUND DATASETS](https://serv.cusp.nyu.edu/projects/urbansounddataset/index.html)
- [Urban Sound Classification, Part 1](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)
- [Urban Sound Classification, Part 2](https://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/)
- [ENVIRONMENTAL SOUND CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS](http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf) ([code](https://github.com/karoldvl/paper-2015-esc-convnet))
