from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
subsampling = 1
xcol = 6
ycol = 1
my_data = genfromtxt(sys.argv[1], delimiter=',')
#print(my_data)
ind=np.arange(0,len(my_data[:,0]),subsampling)
x=my_data[:,xcol][ind]
y=my_data[:,ycol][ind]
# avg_reward=np.sum(x)/len(x)
plt.plot(x,y)
plt.xlabel('Timestamp')
plt.ylabel('Mean Reward')
plt.show()