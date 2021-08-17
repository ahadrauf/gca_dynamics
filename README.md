# GCA Dynamics
Dynamics characterization for a MEMS electrostatic inchworm motor and its gap closing actuators

## Installation
This code just requires `numpy`, `scipy`, and `matplotlib`, although some scripts also involve the 
`scikit-learn.metrics` module. A good script to run first after setup
is `python sim_gca_transient.py` to get a sense for what's going on.

You install the requirements all in one go by calling ```pip install -r requirements.txt```.

## Code Structure
This simulation model is meant to be modular, so you can easily swap between different layouts and/or
processes without changing too much of the actual simulation code.

* `process.py`: the process parameters for a 2-layer SOI process, or any other processes that you
  might come up with. You can import your respective process with 
  ```python 
  from process import SOI
  ```
* `fawn.csv`, `gamma.csv`, etc.: the layout files (named after the layouts of their respective runs).
These files store the drawn dimensions of your model. The code written so far is robust to anything
  after the second comma (so you can write comments there).
  
* `gca.py`: A simulation model for a gap closing actuator. The input is a drawn dimensions filename
    and the initial position. The default initial positions for pull-in and release can be found via
  the functions `GCA.x0_pullin()` and `GCA.x0_release()`.
   - The simulation input is an input array `u = [V, F_external]`. The state is `[x_spine, v_spine]`.
    
* `assembly.py`: Used to package multiple GCA's or other components together. Each component's state
can affect each other, so this modularity can help simulate larger components.
  - `from assembly import AssemblyGCA`: A basic GCA, with no other components
    
* `sim_gca_V_Fext_pullin.py`, etc.: Your simulation files. Because of the structure above, you can
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

## Acknowledgements
This library was inspired by work by Daniel Contreras and Craig Schindler during their Ph.D.
at UC Berkeley under Professor Kristofer Pister. Their Ph.D. dissertations can be found 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-18.html) and 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-73.html), respectively. Many thanks
to them for providing the preliminary MATLAB simulation scripts they used in their dissertations.