import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# TODO: Using OpenCV use BGR plot technique to animate multiple outputs
# In particular animate the changing activation functions and weights

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
  pullData = open("l2_error.csv", "r").read()
  dataArray = pullData.split('\n')
  xar = []
  yar = []
  for eachLine in dataArray:
    if len(eachLine) > 1:
      x,y = eachLine.split(',')
      xar.append(int(x))
      yar.append(float(y))
  ax1.clear()
  ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

