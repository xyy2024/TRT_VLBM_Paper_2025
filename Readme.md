# Code for the TRT-VLBM Paper

We use these codes to do the numerical experiments showed in our paper. The purpose of writing these programs is to apply the model to flows under different conditions as simply as possible.

## Requirements

These are the environment dependencies of these code:
* `python >= 3.13`
* `matplotlib >= 3.9.2`
* `numpy >= 2.2.3`

If you need to output animations, the following dependencies are also required:
* `ffmpeg`
* `openh264` (optional)

## File structure

The correspondence between the file/folder names and their contents is as follows:

### Utility Functions

* `_nproll.py`: this file contains a Python implementation of the Matlab function `circshift`.
* `_systools.py`: this file contains these functions and classes:
* * *func* `printPercent`: Print the progress in the form of a percentage.
* * *class* `error_behavior`: A context class for setting the error handling behavior of NumPy.
* * *class* `cached_property`: Modified cached_property that allows you to register functions that will be called before or after the value of the property has been changed.
* `_plottools.py`: this file contains several functions and classes that is useful when output graphics.

### Main Model

* `d2n5_taylorgreen.py`: base class for solving all conditions, especially for 2D conditions.<br />
This file also contains the setting for period boundary condition, zero outer body force, and initial velocity:

$$\begin{align*}
    u_0(x, y) &= -\cos(2\pi x -0.5\pi) \sin(2\pi y-0.5\pi),\\
    v_0(x, y) &= \sin(2\pi x -0.5\pi) \cos(2\pi y-0.5\pi)
\end{align*}$$

* `d3n7_taylorgreen.py`: base class for solving 3D conditions.<br />
This file also contains the setting for period boundary condition, zero outer body force, and initial velocity:

$$\begin{align*}
    u_0(x, y, z) &= -\cos(2\pi x -0.5\pi) \sin(2\pi y-0.5\pi),\\
    v_0(x, y, z) &= \sin(2\pi x -0.5\pi) \cos(2\pi y-0.5\pi),\\
    w_0(x, y, z) &= 0
\end{align*}$$

### Others
* `flow_pass_cylinder_animation.mp4`: numerical experiment result for flow pass something presented in the form of animation.
* other directories: samples for simulate fluids under other bondary conditions and/or other initial velocities.

## How to simulate fluids under other conditions?

### Step 1. Setup the model.

Write a new subclass of `d2n5_taylorgreen` or `d3n7_taylorgreen` (which can be found at `d2n5_taylorgreen.py` or `d3n7_taylorgreen.py`) and override some method or attributes of the base class according to the condition you want. Such as:

```
from d2n5_taylorgreen import d2n5_taylorgreen
import numpy as np

class sample(d2n5_taylorgreen):
    def init_exact(self):  # Set the initial velocity to zero.
        # This method shall return `u, v`.
        return (np.zeros(self.X) for _ in range(2))

    kf = math.tau
    def get_outerforce(self):   # Set the general source term.
        this = np.zeros((self.Nx, self.Ny, 2))
        this[:,:,0] = np.sin(self.kf*self.Y)
        return this
```

Tips for overriding can be found at `d2n5_taylorgreen.py`. 

### Step 2. Run.

Generate an instance of the new subclass and use one of these method `until_time()` `until_stable()` `until_step()` to start the simulation. 

Then, use one of these method `fig()` `fig_with_error()` to show the result fig. Such as:

```
solver = sample(h = 0.1, nu = 0.1)
solver.until_time(time = 10)
solver.fig()
```

If you want to output animation, use `animation` to start the simulation. The animation of the result will be automatically generated.

## Cite

If our code is useful for your study, please cite our article.
<!-- TODO
### BibTex

-->
