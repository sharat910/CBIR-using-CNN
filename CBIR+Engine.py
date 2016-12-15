
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from retrieve import getvector
import glob
import pickle
import time
from newinput import *


# ## Helper Functions

# In[2]:

def dumptree(tree):
    f = open("bigtree.pickle","w")
    s = pickle.dump(tree,f)
    f.close()


# In[3]:

def undumptree():
    f = open("tree.pickle","r")
    tree = pickle.load(f)
    f.close()
    return tree


# In[4]:

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic


# In[5]:

def view_images(images):

	fig = plt.figure(figsize=(30, 10))

	subplotno=250
	for one_image in images:
		subplotno=subplotno+1
		#one_image = images[img_cnt]



		image_matrix=[]
		for i in range(0,1024):
			temp=[]
			temp.append(one_image[i])
			temp.append(one_image[i+1024])
			temp.append(one_image[i+2*1024])
			image_matrix.append(temp)

		j=0
		farray=[]
		while j<1023:
			array=[]
			for i in range(0, 32):
				array.append(image_matrix[j])
				j=j+1
			farray.append(array)

		ax = fig.add_subplot(subplotno)
		plt.imshow(farray)


	plt.show()


# ## Building the dataset

# In[6]:

def get_train_images():
    filelist = glob.glob("cifar-10-batches-py/d*")
    images = []
    for f in filelist:
        images += list(unpickle(f)['data'])
    return images


# In[7]:

def build(images):
    vectors = [getvector(image)[0][0] for image in images]
    print "here"    
    tree = KDTree(vectors, leaf_size=10) 
    dumptree(tree)


# In[8]:

def query(image):
    view_images(image)
    vector = getvector(image)[0][0]
    tree = undumptree()              
    distances, indices = tree.query(vector, k=5)
    print indices
    print distances
    retrieved_images = [images[i] for i in indices[0]]
    view_images(retrieved_images)    


# In[9]:

images = get_train_images()


# In[10]:

build(images[:100])


# In[20]:

#query([images[240]])


# In[ ]:



