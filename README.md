# GCA Dynamics
Dynamics characterization for a MEMS electrostatic inchworm motor and its gap closing actuators

## Installation Requirements
This code just required numpy, scipy, and matplotlib.

## Code Structure
This simulation model is meant to be modular, so you can easily swap between different layouts and/or
processes without changing too much of the actual simulation code.

* `process.py`: the process parameters for a 2-layer SOI process, or any other processes that you
  might come up with. You can import your respective process with `from process import SOI`.
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



## Acknowledgements
This library was inspired by work by Daniel Contreras and Craig Schindler during their Ph.D.
at UC Berkeley under Professor Kristofer Pister. Their Ph.D. dissertations can be found 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-18.html) and 
[here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-73.html), respectively. Many thanks
to them for providing the preliminary MATLAB simulation scripts they used in their dissertations.