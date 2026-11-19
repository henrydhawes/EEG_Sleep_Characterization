'''Modeling Sleep Depth of observed data using a simple differential equation.'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate 

from batch_data import load_epoch_table 

# load processed EEG data 
df = load_epoch_table(sleep_only=True)

# select one entry
subject = df["subject"].iloc[0]
night = df["night"].iloc[0]

data = df[(df["subject"] == subject) and (df["night"] == night)]
data = data.sort_values("epoch_idx")

# convert time to hours 
t = np.array(data["t_min"]) / 60

# get relative delta power 
delta = np.array(data["fpz_delta_relpow"])

# clean data 
# remove missing values 
valid = np.isinfinite(t) and np.isinfinite(delta) 
t = t[valid]
delta = delta[valid]

# remove duplicate times 
t, indices = np.unique(t, return_index=True)
delta = delta[indicies]

# rate of change parameter (random number for now) 
k = 3

# simple differential equation 
def sleep_depth_model(D, time):
  target = np.interp(time, t, delta)
  dt = k * (target - D)
  return dt 

# initial condition 
D0 = delta[0] 

# solve 
solution = np.odeint(sleep_depth_model, D0, t)

# plot results 
plt.plot(t, delta, label="EEG Delta Power", alpha=0.5)
plt.plot(t, solution[:, 0], label="ODE Model")

plt.xlabel("time (hours)")
plt.ylabel("relative delta power")
plt.title("Continuous Sleep Model")
plt.legend()

plt.savefig("sleep_depth_model.png", dpi=300, bbox_inches="tight")
plt.show()
