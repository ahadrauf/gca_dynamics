"""
Compute the trace length and relevant properties of a path in KLayout
"""

import numpy as np

# Widths = the longest path
# Widths_alternative = the shorter split between the two sides of a GCA finger array (only for HV traces)
widths = 1e-6*np.array([30., 30., 30., 30., 30., 30., 50.])
widths_alternative = 1e-6*np.array([30., 30., 50.])

trajs = ["""-5823.88200	4347.27400
            -5823.88200	4203.54400
            -5717.05200	4203.54400
            -5717.05200	4126.06900
            -5717.05200	4126.06900""",
         """-6434.73500	5132.38300
            -5820.91800	5132.38300
            -5820.91800	4347.27400""",
         """-6436.69900 5119.02500
            -6436.69900 5768.54000""",
         """-6447.12800 5753.54000
            -6360.69900 5753.54000""",
         """-6360.69900 5760.06500
            -6360.69900 6214.85600""",
         """-6368.70600 6199.85600
            -6218.78700 6199.85600""",
         """-6241.64300 6209.47600
            -5446.81500 6209.47600"""]
trajs_alternative = ["""-6360.69900 5753.54000
                        -6218.78700 5753.54000""",
                     """-6211.78700 5758.06500
                        -6211.78700 5958.06500""",
                     """-6241.64300 6209.47600
                        -5446.81500 6209.47600"""]
alternative_split = -3  # the index to split the trajs array by

lengths = []
lengths_alternative = []
for traj in trajs:
    traj = traj.split()
    traj = [np.array([float(traj[2*i]), float(traj[2*i + 1])]) for i in range(len(traj)//2)]
    dist = [np.linalg.norm(traj[i] - traj[i + 1]) for i in range(len(traj) - 1)]
    length = np.sum(dist)
    lengths.append(length*1e-6)
for traj in trajs_alternative:
    traj = traj.split()
    traj = [np.array([float(traj[2*i]), float(traj[2*i + 1])]) for i in range(len(traj)//2)]
    dist = [np.linalg.norm(traj[i] - traj[i + 1]) for i in range(len(traj) - 1)]
    length = np.sum(dist)
    lengths_alternative.append(length*1e-6)

resistivity = 0.1  # Ohm-m
t_SOI = 40e-6
resistances = [resistivity*length/width/t_SOI for length, width in zip(lengths, widths)]
resistances_alternative = [resistivity*length/width for length, width in zip(lengths_alternative, widths_alternative)]
tot_resistance = np.sum(resistances[:alternative_split]) + 1./(1./np.sum(resistances[alternative_split:]) + 1./np.sum(resistances_alternative))

eps0 = 8.85e-12
t_ox = 2e-6
capacitances = [eps0*width*length/t_ox for length, width in zip(lengths, widths)]
capacitances_alternative = [eps0*width*length/t_ox for length, width in zip(lengths_alternative, widths_alternative)]
tot_capacitance = np.sum(capacitances) + np.sum(capacitances_alternative)

print("High Voltage Traces")
print(len(widths), len(lengths))
print(len(widths_alternative), len(lengths_alternative))
print(trajs)
print(lengths, lengths_alternative)
print(resistances, np.sum(resistances), '-->', tot_resistance)
print(capacitances, np.sum(capacitances), '-->', tot_capacitance)



# GND traces
widths = 1e-6*np.array([30., 30., 30., 30.])

trajs = ["""-5780.05600	4357.57400
            -5780.05600	4300.64400
            -5640.03200	4300.64400
            -5640.03200	4126.06900""",
         """-6413.03200	5172.71300
            -5780.05600	5172.71300
            -5780.05600	4349.89400""",
         """-6398.69900 5157.71300
            -6398.69900 5730.54000""",
         """-6391.23100 5715.54000
            -6180.50300 5715.54000"""]

lengths = []
for traj in trajs:
    traj = traj.split()
    traj = [np.array([float(traj[2*i]), float(traj[2*i + 1])]) for i in range(len(traj)//2)]
    dist = [np.linalg.norm(traj[i] - traj[i + 1]) for i in range(len(traj) - 1)]
    length = np.sum(dist)
    lengths.append(length*1e-6)

resistivity = 0.1  # Ohm-m
resistances = [resistivity*length/width/t_SOI for length, width in zip(lengths, widths)]

print("GND Traces")
print(len(widths), len(lengths))
print(trajs)
print(lengths)
print(resistances, np.sum(resistances))
