# parse and visualize the logfile
import numpy as np
import matplotlib.pyplot as plt
import os.path

# load the log file
with open(os.path.dirname(__file__) + "/../log/log.txt", "r") as f:
    lines = f.readlines()

# parse the individual lines, group by stream (train,val,hella)
losses = {"train": [], "val": [], "test": None}
step_list = []
for line in lines:
    if "validation loss" in line:
        step = int(line.strip().split(",")[0].strip().split(" ")[1])
        step_list.append(step)
print(f"Step: {step_list}")

for line in lines:
    if "training loss" in line or "validation loss" in line:
        print(line)
        step = int(line.strip().split(",")[0].strip().split(" ")[1])   
        if step in step_list:
            if "training loss" in line:
                losses["train"].append(float(line.strip().split(",")[1].strip().split(":")[1]))
            elif "validation loss" in line:
                losses["val"].append(float(line.strip().split(",")[1].strip().split(":")[1]))
    elif "Testing loss" in line:
        losses["test"] = float(line.strip().split(":")[1])

# create figure
plt.figure(figsize=(16, 6))
# losses: both train and val
xs = step_list 
ys = losses["train"] # training loss
plt.plot(xs, ys, label=f'train loss')
ys = losses["val"] # validation loss
plt.plot(xs, ys, label=f'val loss')
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.title("Loss")
plt.savefig('loss.png')

test_loss = losses["test"]
print(f"Testing loss: {test_loss}")