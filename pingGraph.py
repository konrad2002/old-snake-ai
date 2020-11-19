import numpy as np
import matplotlib.pyplot as plt

with open('pings.txt') as f:
    lines = f.readlines()

    x = []
    y = []

    for line in lines:
        parts = line.split("$")
        x.append(parts[0])
        y.append(parts[1])

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Ping")    
ax1.set_xlabel('time')
ax1.set_ylabel('ping time')

ax1.plot(x,y, c='r', label='the data')

leg = ax1.legend()

plt.show()