# CODE FOR VLBM PAPER

I use these codes to do the numerical experiments showed in my paper. The purpose of writing these programs is to apply the model to flows under different conditions as simply as possible.

The correspondence between the file names and their contents is as follows:

* `_***.py`: some tools funtion.
* `d2n5_taylorgreen.py`: base class for all models, especially for D2N5 model.
* `d3n7_taylorgreen.py`: base class for D3N7 model.
* `d*n*_***.py`: classes for specific flow.
* `ne***.py`: codes for numerical experiments.
* `Flow_Pass_Something_Animation/*.mp4`: animation figures mentioned in the paper.