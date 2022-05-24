# GCA Dynamics
Dynamics characterization for a MEMS electrostatic inchworm motor and its gap closing actuators

**Publication**
* A. M. Rauf, D. S. Contreras, R. M. Shih, C. B. Schindler, and K. S. J. Pister, “Nonlinear Dynamics of Lateral Electrostatic Gap Closing Actuators for Applications in Inchworm Motors,” Journal of Microelectromechanical Systems, vol. 31, no. 1, pp. 29–36, Feb. 2022, [doi: 10.1109/JMEMS.2021.3130957](https://doi.org/10.1109/JMEMS.2021.3130957).

## Installation
This code just requires `numpy`, `scipy`, and `matplotlib`, although some scripts also involve the 
`scikit-learn.metrics` module. A good script to run first after setup
is `python sim_gca_transient.py` to get a sense for what's going on.

You install the requirements all in one go by calling ```pip install -r requirements.txt```.

**Note for Running This Code in Command Line/CodeOcean**

This code was written in PyCharm, which runs files in their native locations and includes all files in the
overall directory in its system path. This doesn't happen when running this code in command line, so some script
files may require you to (a) navigate to the ``scripts/`` directory before running your code, and
(b) fiddle with the import paths. See the FAQ below for code snippets that you can
try appending to the top of your file to help out with (b).

## Code Structure
This simulation model is meant to be modular, so you can easily swap between different layouts and/or
processes without changing too much of the actual simulation code.

* `process.py`: the process parameters for a 2-layer SOI process, or any other processes that you
  might come up with. You can import your respective process with 
  ```python 
  from process import SOI
  ```
* `layouts/fawn.csv`, `layouts/gamma.csv`, etc.: the layout files (named after the layouts of their respective runs).
These files store the drawn dimensions of your model. The code written so far is robust to anything
  after the second comma (so you can write comments there). See the section below with more details on the required parameters.
  
* `gca.py`: A simulation model for a gap closing actuator. The input is a drawn dimensions filename
    and the initial position. The default initial positions for pull-in and release can be found via
  the functions `GCA.x0_pullin()` and `GCA.x0_release()`.
   - The simulation input is an input array `u = [V, F_external]`. The state is `[x_spine, v_spine]`.
    
* `assembly.py`: Used to package multiple GCA's or other components together. Each component's state
can affect each other, so this modularity can help simulate larger components.
  - `from assembly import AssemblyGCA`: A basic GCA, with no other components
  
*  `scripts/`: All the script files. There are generally 4 types of scripts in this folder:
    - `scripts/calculate_*.py`: These are just helper files for calculating different numbers in the paper.
    - `scripts/plot_*.py`: These are plotting file, which I used to generate the figures in the paper.
    - `scripts/sim_*.py`: These are the core simulation files. Run one of these to simulate a GCA's dynamics.
    - `scripts/test_*.py`: These are another type of helper file, which I used to test different evaluation metrics and the like.
  
* `scripts/sim_gca_V_Fext_pullin.py`, etc.: Your simulation files. Because of the structure above, you can
swap process nodes by changing an import statement or layouts by just swapping which `.csv` file 
  you're reading from. The basic structure of the code is to:
  1. Initialize a model with its drawn dimensions file and specify whether its initial state 
     (e.g. whether it's pulling in or releasing)
  2. Specify the input, e.g. `u = [V, Fext]`
  3. Simulate the assembly with a differential equation solver, like in the code below. You can then
     check whether or not your simulation terminated by checking whether `len(sol.t_events[0]) > 0`:
    
```python
from scipy.integrate import solve_ivp
def sim_gca(model, u, t_span, verbose=False):
    f = lambda t, x: model.dx_dt(t, x, u, verbose=verbose)
    x0 = model.x0()
    terminate_simulation = lambda t, x: model.terminate_simulation(t, x)
    terminate_simulation.terminal = True

    sol = solve_ivp(f, t_span, x0, events=[terminate_simulation], dense_output=True, max_step=0.5e-6)
    return sol
```

Note that the `max_step` parameter in `solve_ivp()` is actually quite important - generally,
in Python's implementation of `solve_ivp()`, for small time scales like what we're working with
the final time recorded by the `terminate_simulation` condition will be some multiple of `max_step`.
Thus, for production code it's generally good to have low values for `max_step` (e.g. 0.1e-6), but for 
quick prototyping it's sufficient to have larger values (e.g. 0.25e-6, 0.5e-6).

## FAQ's
* I get the error ```gca.py:149: RuntimeWarning: invalid value encountered in sqrt``` or ```gca.py:177: IntegrationWarning: The occurrence of roundoff error is detected, which prevents 
  the requested tolerance from being achieved.  The error may be 
  underestimated.``` when running my code and using Fes_calc2 (i.e. choosing calc_method=2 as an option
  for Fes() in gca.py).
    * This is likely because the time step of the integration is too large. You can change this in the
    max_step option in solve_ivp() (e.g. see the sim_gca() function in sim_gca_V_fingerL_pullin_release.py
      for reference. Note that not all integral solvers in Scipy accept the max_step argument - the default
      RK45 method has been found to work pretty well). The error is caused when the gap between fingers
      becomes negative (i.e. pull-in happens, but the integration routine jumps over the 
      terminate_simulation boundary condition instead of converging to it).
      
* I get the warning ```mio.py:226: MatReadWarning: Duplicate variable name "None" in stream - replacing previous with new
Consider mio5.varmats_from_mat to split file into single variable files
  matfile_dict = MR.get_variables(variable_names)```
  * Probably not an issue? This is caused by some file formatting issue in how I'm reading the data files (in the ```/data```
    folder). I'm not sure exactly how to fix this, to be honest - fiddling with the import settings should
  be enough, but it reads the data correctly so who cares?
    
* I get ```ModuleNotFoundError``` errors when trying to run scripts in command line
  * See the note at the top about running this code in command line. Add the following code to the beginning of your script file, making sure to change the part in brackets to your 
  specific installation
  ```python
  import os
  file_location = os.path.abspath(os.path.dirname( __file__))
  dir_location = os.path.abspath(os.path.join(file_location, '..'))
  import sys
  sys.path.append(file_location)  # or just the path to the script's folder, generally speaking
  sys.path.append(dir_location)   # or just the path to the gca_dynamics folder, generally speaking
  ```
  
## Layout and Process Files
Unfortunately in hindsight, many of the parameters in the layout file somewhat differ from the paper. Below is a description, both 
textually and graphically, of the important parameters when defining the layout and process files.

###Layout Files

Note: All distances are written in the layout file as drawn (don't include undercut!)
```
name, value
gf, 4.83e-6  # front gap (called x_0 in the paper)
gb, 7.751e-6  # back gap (called x_b in the paper)
x_GCA, 3.83e-6  # distance the spine has to travel before hitting the gap stop (x0 - xf in the paper)
supportW, 3e-6  # width of the support beams (called w_spr in the paper)
supportL, 240.851e-6  # length of the support beams (called L_spr in the paper) 
Nfing, 70  # number of GCA fingers (called N in the paper)
fingerW, 5.005e-6  # width of GCA fingers (called wf in the paper)
fingerL, 76.472e-6  # overlap length of the GCA fingers (called Lol in the paper)
fingerL_buffer, 10e-6  # extra length at the base of the GCA fingers but which doesn't overlap adjacent fingers (called L - Lol in the paper)
spineW, 20e-6  # width of the spine (called w_spine in the paper) 
spineL, 860e-6  # length of the spine (called L_spine in the paper)
etch_hole_size, 8e-6  # size of etch hole squares in the spine (not in paper, see figure below)
etch_hole_spacing, 6e-6  # spacing between etch hole squares in the spine (not in paper, see figure below)
gapstopW, 10e-6  # width of the gap stop jutting out from either side of the spine (not in paper, see figure below)
gapstopL_half, 45e-6  # length of the gap stop jutting out from either side of the spine (not in paper, see figure below). To avoid double-counting area, this is only the length from the side of the spine to the end of the gapstop.
anchored_electrodeW, 787.934e-6  # this was used for an older version of the GCA schematic, can be deleted in more recent versions
anchored_electrodeL, 50e-6  # this was used for an older version of the GCA schematic, can be deleted in more recent versions
```

![Layout file dimension diagram](figures/layout_file_dimension_diagram.png)

### Process Files
See the file `process.py` for descriptions of all the process parameters when fabricating and testing the devices. Some
parameters, such as the undercut, unfortunately needs to be calibrated for your fabrication process (although you can
also fit the undercut via the `_undercut.py` scripts in the `scripts/` folder).


## Acknowledgements
This library originated from work by Daniel Contreras (Ph.D.), Craig Schindler (Ph.D.), and Ryan Shih (M.S.)
at UC Berkeley under Professor Kristofer Pister. Daniel's and Craig's Ph.D. dissertations can be found 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-18.html) and 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-73.html), respectively. Many thanks
to them for providing the preliminary MATLAB simulation scripts they used in their dissertations.