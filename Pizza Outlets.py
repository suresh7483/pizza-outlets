# Importing essential Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

print('Start')
print("Enter the number of Center you want in the data:",end=" ")
K = int(input())

X,y = make_blobs(n_samples=1000,n_features=2,centers=2,random_state=5)

plt.figure(0)
plt.scatter(X[:,0],X[:,1])
plt.figure(1)

class Kmeans():

	def __init__(self,X,K):
		self.K = K
		self.data = X

	def distance(self,v1,v2):
    	
		return np.sqrt(np.sum((v1 - v2)**2))


	def make_cluster(self):
	    colors = ['red','blue','pink','yellow','violet','green','cyan','magenta','black','white']
	    self.clusters = {}
	    for kx in range(self.K):
	        
	        center = X[np.random.randint(0,self.K + 1)]
	        color = colors[kx]

	        points = []
	        cluster = {

	            'points' : points,
	            'center' : center,
	            'color' : color
	        }
	        self.clusters[kx] = cluster

	def AssignPoints(self):
	    
	    for kx in range(self.K):
	        self.clusters[kx]['points'] = []
	        
	    for ix in range(self.data.shape[0]):

	        curr_point = self.data[ix]
	        dist = []
	        for kx in range(self.K):

	            d = self.distance(curr_point,self.clusters[kx]['center'])
	            dist.append(d)

	        current_cluster = np.argmin(dist)
	        self.clusters[current_cluster]['points'].append(curr_point)

	def UpdateCenter(self):
	    
	    old_centers = []
	    for kx in range(self.K):
	        
	        pts = np.array(self.clusters[kx]['points'])
	        
	        if pts.shape[0] > 0:
	        
	            new_center = pts.mean(axis=0)
	            old_centers.append(self.clusters[kx]['center'])
	            self.clusters[kx]['center'] = new_center

	    return old_centers

	def plot_cluster(self):
	    

	    for kx in range(self.K):
	        
	        pts = np.array(self.clusters[kx]['points'])
	        

	        plt.scatter(pts[:,0],pts[:,1],color=self.clusters[kx]['color'])

	        
	        center = self.clusters[kx]['center']
	        plt.scatter(center[0],center[1],color='black',marker="*")


	def Predict(self,epochs = 30,min_value=0.01):
	    
	    self.make_cluster()
	    self.AssignPoints()
	    count = 0
	    
	    while count < epochs:        
	        is_saturated = True
	        old_centers = self.UpdateCenter()

	        for kx in range(self.K):
	            new_center = self.clusters[kx]['center']
	            old_center = old_centers[kx]
	            dist = self.distance(old_center,new_center)
	            
	            if dist > min_value:
	                is_saturated = False
	                break

	        if is_saturated is False:
	            self.AssignPoints()
	        else:
	            break
	        
	        count += 1
	    new_centers = []

	    for kx in range(self.K):
	    	new_centers.append(self.clusters[kx]['center'])

	    return new_centers


classifier = Kmeans(X,K)

new_centers = classifier.Predict()
for i in new_centers:
	print(i)

classifier.plot_cluster()

plt.show()

print("End")