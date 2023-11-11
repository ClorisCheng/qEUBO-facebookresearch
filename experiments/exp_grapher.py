#%%
import os
import numpy as np
import matplotlib.pyplot as plt


os.chdir('/home/ec2-user/projects/qEUBO-facebookresearch/experiments')

# Load data
last_trial = 3
root_dir = 'results/ackley'

fns = ["max_obj_vals_within_queries", "obj_vals_at_max_post_mean"]
algos = ["qeubo_3", "qeubo_3_mclike"]

max_obj_vals = {}
obj_vals_at_max = {}

for algo in algos:
    max_obj_vals[algo] = []
    obj_vals_at_max[algo] = []
    path = os.path.join(root_dir, algo)
    for fn in fns:
        for trial in range(1, last_trial + 1):
            file_path = os.path.join(path, f"{fn}_{trial}.txt")
            if fn == "max_obj_vals_within_queries":
                max_obj_vals[algo].append(np.loadtxt(file_path))
            elif fn == "obj_vals_at_max_post_mean":
                obj_vals_at_max[algo].append(np.loadtxt(file_path))
    
    # stack so that the shape is (num_trials, num_queries)
    max_obj_vals[algo] = np.stack(max_obj_vals[algo])
    obj_vals_at_max[algo] = np.stack(obj_vals_at_max[algo])


#%%
# Plot

# Plot max objective values
fig = plt.figure()
ax = fig.add_subplot(111)
for algo in algos:
    mean = max_obj_vals[algo].mean(axis=0)
    std = max_obj_vals[algo].std(axis=0)
    ax.plot(mean, label=algo)
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

ax.set_title("Max objective values")
ax.set_xlabel("Queries")
ax.set_ylabel("Objective value")
ax.legend()
plt.show()


# %%
# plot objective values at max posterior mean
fig = plt.figure()
ax = fig.add_subplot(111)
for algo in algos:
    mean = obj_vals_at_max[algo].mean(axis=0)
    std = obj_vals_at_max[algo].std(axis=0)
    ax.plot(mean, label=algo)
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

ax.set_title("Objective values at max posterior mean")
ax.set_xlabel("Queries")
ax.set_ylabel("Objective value")
ax.legend()
plt.show()
# %%
