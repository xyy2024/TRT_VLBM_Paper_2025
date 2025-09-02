# Code for the TRT-VLBM Paper "Two-relaxation-time Vectorial Lattice Boltzmann Method for Solving Incompressible Navier–Stokes Equations" 

This repository contains the source codes used to perform the numerical experiments presented in our paper:

> **Bangwei She, Yuyang Xu, Jin Zhao.**  
> *Two-relaxation-time Vectorial Lattice Boltzmann Method for Solving Incompressible Navier–Stokes Equations*, arXiv, 2025.  
> [Link / DOI once available]

The primary goal of these programs is to provide a straightforward framework for applying the proposed model to simulate fluid flows under various conditions.

---

## **Repository Structure**

The correspondence between file names and their contents is as follows:

- **`_***.py`** – Utility functions.  
- **`d2n5_taylorgreen.py`** – Base class for all models, specifically for the D2N5 model.  
- **`d3n7_taylorgreen.py`** – Base class for the D3N7 model.  
- **`d*n*_***.py`** – Classes implementing specific flow configurations (examples of model customization).  
- **`ne***.py`** – Scripts for numerical experiments.  
- **`Flow_Pass_Something_Animation/*.mp4`** – Animation videos referenced in the paper.

---

## **Usage**

### **How to Simulate Different Flow Configurations**

1. **Inherit a Base Class**  
   - Use `d2n5_taylorgreen.py` (for D2N5) or `d3n7_taylorgreen.py` (for D3N7) as the base class.  
   - Override selected methods or attributes to adapt the model to your desired configuration.  
   - Detailed comments in the base classes provide guidance for overriding.

2. **Example Implementations**  
   - Files matching the pattern `d*n*_***.py` serve as examples demonstrating how to customize the base classes.
   - Files matching `ne***.py` provide examples of how to run simulations using these customized classes.

3. **Animation Outputs**  
   - The `Flow_Pass_Something_Animation/*.mp4` folder contains example animation results from our simulations.

---

## **Citation**

If you use these codes in your research, please cite our paper:

```bibtex
@article{SheXUZhao2025TRT,
  author  = {Bangwei She and Yuyang Xu and Jin Zhao},
  title   = {Two-relaxation-time Vectorial Lattice Boltzmann Method for Solving Incompressible Navier–Stokes Equations},
  journal = {arXiv},
  volume  = {1},
  number  = {1},
  pages   = {1},
  year    = {2025},
  doi     = {},
  url     = {},
  eprint  = {}
}
```

---

## **Notes**

- The code base is written in Python and structured for ease of modification and extension.  
- For developers intending to extend the framework, following the examples in the `d*n*_***.py` and `ne***.py` scripts is recommended.  
- The repository is aimed at researchers in computational fluid dynamics, especially those working with the lattice Boltzmann method.

---
