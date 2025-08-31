# CODE FOR VLBM PAPER

I use these codes to do the numerical experiments showed in my paper. The purpose of writing these programs is to apply the model to flows under different conditions as simply as possible.

The correspondence between the file names and their contents is as follows:

* `_***.py`: some tools funtion.
* `d2n5_taylorgreen.py`: base class for all models, especially for D2N5 model.
* `d3n7_taylorgreen.py`: base class for D3N7 model.
* `d*n*_***.py`: classes for specific flow.
* `ne***.py`: codes for numerical experiments.
* `Flow_Pass_Something_Animation/*.mp4`: animation figures mentioned in the paper.

## Usage: how to simulate other situations?
Simply inherit the base class in `d2n5_taylorgreen.py` or `d3n7_taylorgreen.py` and override some
method or attributes of the base class. Tips for overriding can be found in the comments of the base class.

The classes in `d*n*_***.py` can be regarded as examples or samples of how to override. And the codes in 
`ne***.py` can be regarded as samples of using these classes for fluid simulation.