import get_datasets as gd


import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import re
#PANDAS
from pandas.core.frame import DataFrame
import pandas as pd

import igraph as ig
from enum import Enum
#SKLEARN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#PLOTLY
import plotly.tools 
import plotly.plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from asn1crypto.core import InstanceOf
import sklearn

plotly.tools.set_credentials_file(username='isonettv', api_key='2Lg1USMkZAHONqo82eMG')

'''
The mode used in a cycle. Can be one of:
    TRAIN: to use when training the model
    TEST: to use when testing the model
'''
Mode = Enum("Mode","TRAIN TEST PRED")


'''
Neighborhood function h_ij to be used. Can be one of:
    GAUSSIAN: Uses a gaussian with decay equal to self.sigma on the neighborhood
    CONSTANT: 1 if in neighborhood, 0 otherwise
'''
NFunc = Enum("NFunc","GAUSSIAN CONSTANT")


'''
Initialization of the initial points.Can be one of:
    RANDOM: Initializes each points randomly inside (-self.init_maxval,+self.init_maxval)
    PCAINIT: Uses the 2 principal components to unfold a grid
'''
InitMode = Enum("InitMode","RANDOM PCAINIT")

'''
Configuration for the plot operation.Can be one of:
    CLASS_COLOR: Shows the class attr. of unlabeled points through color
    CLASS_NOCOLOR: All unlabeled points have the same color
'''

PlotMode = Enum("PlotMode","CLASS_COLOR CLASS_NOCOLOR")
'''
Configuration for grid type. Can be one of:
    RECTANGLE:    Use rectangular grid
    HEXAGON:    Use hexagonal grid
'''
GridMode = Enum("GridMode","RECTANGLE HEXAGON")




class som(object):


    '''
    Creates a Self-Organizing Map (SOM).
        It has a couple of parameters to be selected by using the "args" object. These include:
            
            self.N1: Number of nodes in each row of the grid
            
            self.N2: Number of nodes in each column of the grid
            
            self.eps_i, self.eps_f: initial and final values of epsilon. The current value of
                epsilon is given by self.eps_i * (self.eps_f/self.eps_i)^p, where p is percent
                of completed iterations.
            self.sigma_i, self.sigma_f: initial and final values of sigma. The current value of
                epsilon is given by self.eps_i * (self.eps_f/self.eps_i)^p, where p is percent
                of completed iterations.
                
            self.ntype: Neighborhood type
            self.plotmode: which plot to make
            self.initmode: how to initialize grid 
            self.gridmode: which type of grid 
            
            self.ds_name: Name of dataset (iris,isolet,wine,grid)  
                
            
    '''
    
    

    ''' Initializes the Growing Neural Gas Network
        @param sess: the current session
        @param args: object containing arguments (see main.py)
    ''' 
    def __init__(self, sess, args): # Parameters
        
        self.sess = sess
        
        #TODO: use arguments to get parameters 
        #Number of nodes in each row of the grid
        self.N1 = 10 
        #Number of nodes in each column of the grid
        self.N2 = 10 
        
        #Initial and final values of epsilon
        self.eps_i = 0.5
        self.eps_f = 0.005
        
        #Initial and final values of sigma
        self.sigma_i = 3
        self.sigma_f = 0.1
        
        #Neighborhood type
        self.ntype = NFunc["GAUSSIAN"]

        #Which plot to make
        self.plotmode = PlotMode["CLASS_COLOR"]
        
        #Grid Mode
        self.gridmode = GridMode["HEXAGON"]
        
        self.nsize = 1
        
        #Which way to initialize points
        self.initmode = InitMode["PCAINIT"]
        #interval (for random initialization only)
        self.init_maxval = 0.1
        
        #dataset chosen
        self.ds_name = args.dataset
        
        #Total number of iterations
        self.n_iter = 10000
        #Number of iterations between plot op
        self.plot_iter = 100
        
        self.characteristics_dict = \
            {"dataset":str(self.ds_name),
             "num_iter":self.n_iter,
            "N1":self.N1,
             "N2":self.N2,
             "eps_i":self.eps_i,
             "eps_f":self.eps_f,
             "sigma_i":self.sigma_i,
             "ntype":self.ntype.name,
             "initmode":self.initmode.name,
             "gridmode":self.gridmode.name                
            }
        
        #Get datasets
        if args.dataset == "isolet":
            temp = gd.get_isolet(test_subset= args.cv)
        elif args.dataset == "iris":
            temp = gd.get_iris(args.cv)
        elif args.dataset == "wine":
            temp = gd.get_wine(args.cv)
        elif args.dataset == "grid" or args.dataset == "box":
            temp = gd.get_grid()
        else:
            raise ValueError("Bad dataset name")
        
        
        #Create Dataset
        if  isinstance(temp,dict) and 'train' in temp.keys():
            self.ds = temp["train"].concatenate(temp["test"])
        else:
            self.ds = temp
        
        #Store number of dataset elements and input dimension
        self.ds_size = self.getNumElementsOfDataset(self.ds)
        self.ds_inputdim = self.getInputShapeOfDataset(self.ds)
        
        #Normalize dataset
        temp = self.normalizedDataset(self.ds)
        self.ds = temp["dataset"]
        df_x_normalized = temp["df_x"]
        self.Y = temp["df_y"]
        
        
        #Get PCA for dataset
        print("Generating PCA for further plotting...")
        self.pca = PCA(n_components=3)
        self.input_pca = self.pca.fit_transform(df_x_normalized)
        self.input_pca_scatter = self.inputScatter3D()
        self.input_pca_maxval = -np.sort(-np.abs(np.reshape(self.input_pca,[-1])),axis=0)[5]
        print("Done!")
        print("Dimensionality of Y:{}",self.ds_inputdim)
        
        
        
        self.ds = self.ds.shuffle(buffer_size=10000).repeat()
        
        if args.dataset == "box":
            self.ds = gd.get_box()
        
        #Now generate iterators for dataset
        self.iterator_ds = self.ds.make_initializable_iterator()
        self.iter_next = self.iterator_ds.get_next()
        
        
        
        # tf Graph input
        self.X_placeholder = tf.placeholder("float", [self.ds_inputdim])
        self.W_placeholder = tf.placeholder("float", [self.N1*self.N2,self.ds_inputdim])
        self.nborhood_size_placeholder = tf.placeholder("int32", [])
        self.sigma_placeholder = tf.placeholder("float", [])
        self.eps_placeholder = tf.placeholder("float", [])
        
        print("Initializing graph and global vars...")
        self.init_graph()
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        print("Done!")
        
    ''' Transforms dataset back to dataframe'''    
    def getDatasetAsDF(self,dataset):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        num_elems = 0
        while True:
            try:
                x, y = self.sess.run([next_element["X"], next_element["Y"]])
                
                if num_elems == 0:
                    df_x = pd.DataFrame(0,index = np.arange(self.getNumElementsOfDataset(dataset)),\
                      columns = np.arange(x.shape[0])
                    )
                    print(y)
                    df_y = pd.DataFrame(0,index = np.arange(self.getNumElementsOfDataset(dataset)),\
                      columns = np.arange(y.shape[0])
                    )
                    
                df_x.iloc[num_elems,:] = x
                df_y.iloc[num_elems,:] = y
    
                num_elems += 1
            except tf.errors.OutOfRangeError:
                break
        return({"df_x":df_x,"df_y":df_y}) 
        
    ''' Returns the total number of elements of a dataset
        @param dataset: the given dataset
        @return: total number of elements
    '''
    def getNumElementsOfDataset(self,dataset):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        num_elems = 0
        while True:
            try:
                self.sess.run(next_element)
                num_elems += 1
            except tf.errors.OutOfRangeError:
                break
        return num_elems
            
    ''' Returns the dimensionality of the first element of dataset
        @param dataset: the given dataset
        @return: total number of elements
    '''
    def getInputShapeOfDataset(self,dataset):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        d = None
        try:
            self.sess.run(next_element)
            d = next_element["X"].shape[0]
        except tf.errors.OutOfRangeError:
            return d
        return int(d)
    
    
    
    ''' Returns the normalized version of a given dataset
        @param dataset: the given dataset, such that each element returns an "X" and "Y"
        @return: dict, with keys
                    "df_x": normalized elements,
                    "df_y": corresponding class attr.,
                    "dataset": normalized dataset ("X" and "Y")
    '''
    def normalizedDataset(self,dataset):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        num_elems = 0
        while True:
            try:
                x, y = self.sess.run([next_element["X"], next_element["Y"]])
                
                if num_elems == 0:
                    df_x = pd.DataFrame(0,index = np.arange(self.getNumElementsOfDataset(dataset)),\
                      columns = np.arange(x.shape[0])
                    )
                    print(y)
                    df_y = pd.DataFrame(0,index = np.arange(self.getNumElementsOfDataset(dataset)),\
                      columns = np.arange(y.shape[0])
                    )
                    
                df_x.iloc[num_elems,:] = x
                df_y.iloc[num_elems,:] = y

                num_elems += 1
            except tf.errors.OutOfRangeError:
                break
        df_x = StandardScaler().fit_transform(df_x)
        print(df_y)
        return({"df_x":  df_x,
                "df_y":  df_y,
                "dataset": tf.data.Dataset.from_tensor_slices({"X":df_x,"Y":df_y}) \
                })
        
        
       
    
    ''' Initializes the SOM graph
    '''
    def init_graph(self):
        #initial topology
        self.g = ig.Graph()
        self.g.add_vertices(self.N1*self.N2)

        
        incr = [(0,1),(0,-1),(1,0),(-1,0)]
        def isvalid(x):
            return(x[0] >= 0 and x[0] < self.N1 and\
                   x[1] >= 0 and x[1] < self.N2) 
        def toOneDim(x):
            return x[0]*self.N2 + x[1]
        def sum_tuple(x,y):
            return(tuple(sum(pair) for pair in zip(x,y)))
        edges = []
        
        #Add edges
        for i in np.arange(self.N1):
            for j in np.arange(self.N2):
                curr = (i,j)
                self.g.vs[toOneDim(curr)]["i"] = i
                self.g.vs[toOneDim(curr)]["j"] = j
                
                if self.gridmode.name == "RECTANGLE":
                    incr = [(0,1),(0,-1),(1,0),(-1,0)]
                else:
                    if i % 2 == 0:
                        incr = [(0,1),(0,-1),(-1,-1),(-1,0),(1,-1),(1,0)]
                    else:
                        incr = [(0,1),(0,-1),(-1,1),(-1,0),(1,1),(1,0)]
                
                nbors = list(map(lambda x: sum_tuple(x,curr),incr))
                nbors_exist = list(map(lambda x: isvalid(x),nbors))
                
                for n in np.arange(len(nbors)):
                    if nbors_exist[n]:
                        edges += [(toOneDim(curr), toOneDim(nbors[n]))]
                        print(str(curr) + "->" + str(nbors[n]) )
        self.g.add_edges(edges)
        
        self.g.es["age"] = 0
        
        #self.ID: maps index of each node to its corresponding position tuple (line, col) 
        self.ID = np.array(list(map(lambda x: [self.g.vs[x]["i"],self.g.vs[x]["j"]],\
                               np.arange(self.N1*self.N2))),dtype=np.int32 )
        self.ID = np.reshape(self.ID,(-1,2))
        

    def build_model(self):
        X = self.X_placeholder
        W = self.W_placeholder
        nsize = self.nborhood_size_placeholder
        sigma = self.sigma_placeholder
        eps = self.eps_placeholder
        
        #Step 3: Calculate dist_vecs (vector and magnitude)
        self.dist_vecs = tf.map_fn(lambda w: X - w,W)
        self.squared_distances = tf.map_fn(lambda w: 2.0*tf.nn.l2_loss(X - w),W)
        
        #Step 4:Calculate 2 best 
        self.s = tf.math.top_k(-self.squared_distances,k=2) 
        #1D Index of s1_2d,s2_2d
        self.s1_1d = self.s.indices[0]
        self.s2_1d = self.s.indices[1]
        #2d Index of s1_2d,s2_2d
        self.s1_2d = tf.gather(self.ID,self.s1_1d)
        self.s2_2d = tf.gather(self.ID,self.s2_1d)
        
        
        #Step 5: Calculate l1 distances
        self.l1 = tf.map_fn(lambda x: tf.norm(x-self.s1_2d,ord=1),self.ID)
        self.mask = tf.reshape(tf.where(self.l1 <= nsize),\
                                        tf.convert_to_tensor([-1]))

        #Step 6: Calculate neighborhood function values
        if self.ntype.name == "GAUSSIAN":
            self.h = tf.exp(-tf.square(tf.cast(self.l1,dtype=tf.float32))\
                        /(2.0*sigma*sigma))
        elif self.ntype.name == "CONSTANT":
            self.h = tf.reshape(tf.where(self.l1 <= nsize,x=tf.ones(self.l1.shape),\
                                     y=tf.zeros(self.l1.shape)),
                                        tf.convert_to_tensor([-1]))
        else:
            raise ValueError("unknown self.ntype")
        
        #Step 6: Update W
        self.W_new = W + eps*tf.matmul(tf.diag(self.h), self.dist_vecs)
       
        
    def cycle(self, current_iter, mode = Mode["TRAIN"]):
        #Iteration numbers as floats
        current_iter_f = float(current_iter)
        n_iter_f = float(self.n_iter)
        
        nxt = self.sess.run(self.iter_next)["X"]
        
        
        #Get current epsilon and theta
        eps = self.eps_i * np.power(self.eps_f/self.eps_i,current_iter_f/n_iter_f)
        sigma = self.sigma_i * np.power(self.sigma_f/self.sigma_i,current_iter_f/n_iter_f)
        
        print("Iteration {} - sigma {} - epsilon {}".format(current_iter,sigma,eps))
        
        #Get vector distance, square distance
        self.dist_vecs = np.array(list(map(lambda w: nxt - w,self.W)))
        self.squared_distances = np.array(list(map(lambda w: np.linalg.norm(nxt - w,ord=2),self.W)))

        
        if mode.name == "TEST":   
            #Get first and second activation
            top_2 = np.argsort(self.squared_distances)[0:2]
            self.s1_1d = top_2[0]
            self.s2_1d = top_2[1]
            self.s1_2d = self.ID[self.s1_1d]
            self.s2_2d = self.ID[self.s2_1d]
            #Get topographic error
            if (self.s2_1d in self.g.neighbors(self.s1_1d)):
                topographic_error = 0
            else:
                topographic_error = 1
            #Get quantization error
            quantization_error = (self.squared_distances[self.s1_1d])     
        else:
            self.s1_1d = np.argmin(self.squared_distances,axis=-1)
            self.s1_2d = self.ID[self.s1_1d]
        
        
        
        self.l1 = np.array(list(map(lambda x: np.linalg.norm(x - self.s1_2d,ord=1),self.ID)))
        self.mask = np.reshape(np.where(self.l1 <=  1.38739 * sigma),[-1])
        

        squared_l1 = -np.square(self.l1.astype(np.float32))
        if self.ntype.name == "GAUSSIAN":
            self.h = np.exp(squared_l1 /(2.0*sigma*sigma))
        elif self.ntype.name == "CONSTANT":
            self.h = np.reshape(np.where(self.l1 <= 1.38739 * sigma,1,0),[-1])
        
        for i in np.arange(self.N1*self.N2):
            self.W[i,:] += eps * self.h[i] * self.dist_vecs[i,:]
            
        if mode.name == "TEST":   
            return ({"topographic_error":topographic_error,\
                     "quantization_error":quantization_error})
            
    def train(self):
        #Run initializer
        self.sess.run(self.init)
        self.sess.run(self.iterator_ds.initializer)
        if self.initmode.name == "PCAINIT":
            if self.ds_inputdim < 2:
                raise ValueError("uniform init needs dim input >= 2")
            self.W = np.zeros([self.N1*self.N2,3])
            print(self.W.shape)
            self.W[:,0:2] = np.reshape(self.ID,[self.N1*self.N2,2])
            if self.gridmode.name == "HEXAGON":
                print(list(map(lambda x: (x//self.N2%2==0),np.arange(self.W.shape[0]))))
                self.W[list(map(lambda x: (x//self.N2%2==0),np.arange(self.W.shape[0])))\
                       ,1] -= 0.5
            
            print(self.W.shape)
            self.W = np.matmul(self.W,self.pca.components_)
            print(self.W.shape)
            self.W = StandardScaler().fit_transform(self.W)
            print(self.W.shape)
        else:
            self.W =  self.sess.run(tf.random.uniform([self.N1*self.N2,self.ds_inputdim],\
                                         dtype=tf.float32))
            self.W = StandardScaler().fit_transform(self.W)
       
    #BEGIN Training
        for current_iter in np.arange(self.n_iter):
            
            self.cycle(current_iter)
                     
            if  current_iter % self.plot_iter == 0:
                self.prettygraph(current_iter,mask=self.mask)
         
        self.prettygraph(self.n_iter,mask=self.mask,online=True)
    #END Training
    #BEGIN Testing  
    
        self.sess.run(self.iterator_ds.initializer)
        topographic_error = 0
        quantization_error = 0
        chosen_Mat = np.zeros((self.N1,self.N2))
        
        for current_iter in np.arange(self.ds_size):
            cycl = self.cycle(current_iter,mode=Mode["TEST"])
            topographic_error += cycl["topographic_error"] 
            quantization_error += cycl["quantization_error"]
            chosen_Mat[self.s1_2d[0],self.s1_2d[1]] += 1
        
             
        topographic_error = topographic_error / self.ds_size
        quantization_error =  quantization_error / self.ds_size
        
        #Generate U-Matrix
        U_Mat = np.zeros((self.N1,self.N2))
        for i in np.arange(self.N1):
            for j in np.arange(self.N2):
                vert_pos = self.W[i * self.N2 + j]
                nbors = self.g.neighbors(i * self.N2 + j)
                d = np.sum(\
                           list(map(lambda x: np.linalg.norm(self.W[x] - vert_pos)\
                                    ,nbors)))    
                U_Mat[i,j] = d
        
        
        print("Quantization Error:{}".format(quantization_error))
        print("Topographic Error:{}".format(topographic_error))
        
        print(np.array(self.characteristics_dict.keys()) )
        
        df_keys = list(self.characteristics_dict.keys()) +\
                     ["quantization_error","topographic_error"]
        df_vals =  list(self.characteristics_dict.values()) +\
                                 [quantization_error,topographic_error]
        #df_vals = [str(x) for x in  df_vals]
        
        print(df_keys)
        print(df_vals)
        print(len(df_keys))
        print(len(df_vals))
        if os.path.exists("runs.csv"):
            print("CSV exists")
            df = pd.read_csv("runs.csv",header=0)
            df_new = pd.DataFrame(columns=df_keys,index=np.arange(1))
            df_new.iloc[0,:] = df_vals
            df = df.append(df_new,ignore_index=True)
            
        else:
            print("CSV created")
            df = pd.DataFrame(columns=df_keys,index=np.arange(1))
            df.iloc[0,:] = df_vals
        df.to_csv("runs.csv",index=False)
    def inputScatter3D(self):
        Xn = self.input_pca[:,0]
        Yn = self.input_pca[:,1]
        Zn = self.input_pca[:,2]
        
        
        
        Y = self.sess.run(tf.cast(tf.argmax(self.Y,axis=-1),dtype=tf.int32) )
        Y = [int(x) for x in Y]
        num_class = len(Y)
        pal = ig.ClusterColoringPalette(num_class)

        if self.plotmode.name == "CLASS_COLOR":
            col = pal.get_many(Y)
            siz = 2
        else:
            col = "green"
            siz = 1.5
        trace0=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='input',
                       marker=dict(symbol='circle',
                                     size=siz,
                                     color=col,
                                     line=dict(color='rgb(50,50,50)', width=0.25)
                                     ),
                       text="",
                       hoverinfo='text'
                       )
        return(trace0)    
    def prettygraph(self,iter_number, mask,online = False):
        
        trace0 = self.input_pca_scatter
        W = self.pca.transform(self.W)
        
        Xn=W[:,0]# x-coordinates of nodes
        Yn=W[:,1] # y-coordinates
        if self.ds_name in ["box","grid"]:    
            Zn=self.W[:,2] # z-coordinates
        else:
            Zn=W[:,2] # z-coordinates
            
        edge_colors = []
        Xe=[]
        Ye=[]
        Ze=[]
        
        num_pallete = 1000
        for e in self.g.get_edgelist():
            #col  = self.g.es.find(_between=((e[0],), (e[1],)),)["age"]
            #col = float(col)/float(1)
            #col = min(num_pallete-1, int(num_pallete  * col))
            #edge_colors += [col,col]
            
            Xe+=[W[e[0],0],W[e[1],0],None]# x-coordinates of edge ends
            Ye+=[W[e[0],1],W[e[1],1],None]# y-coordinates of edge ends
            Ze+=[W[e[0],2],W[e[1],2],None]# z-coordinates of edge ends
           
        
        #Create Scaling for edges based on Age
        
        pal_V = ig.GradientPalette("blue", "black", num_pallete)
        pal_E = ig.GradientPalette("black", "white", num_pallete)
        
        v_colors = ["orange" for a in np.arange(self.g.vcount())]
        
        for v in mask:
            v_colors[v] = "yellow"
        


         
        trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=dict(color="black",
                                  width=3),
                       hoverinfo='none'
                       )
        
        reference_vec_text = ["m" + str(x) for x in np.arange(self.W.shape[0])]
        trace2=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='reference_vectors',
                       marker=dict(symbol='square',
                                     size=6,
                                     color=v_colors,
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=reference_vec_text,
                       hoverinfo='text'
                       )
        
        
        axis=dict(showbackground=False,
                  showline=True,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=True,
                  title='',
                  range = [-self.input_pca_maxval-1,1+self.input_pca_maxval]
                  )
        
        layout = go.Layout(
                 title="Visualization of SOM",
                 width=1000,
                 height=1000,
                 showlegend=False,
                 scene=dict(
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     zaxis=dict(axis),
                ),
             margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=[
                   dict(
                   showarrow=False,
                    text="Data source:</a>",
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                    size=14
                    )
                    )
                ],    )       
        data=[trace1, trace2, trace0]
        fig=go.Figure(data=data, layout=layout)
        print("Plotting graph...")
        if online:
            try:
                py.iplot(fig)
            except plotly.exceptions.PlotlyRequestError:
                print("Warning: Could not plot online")
        pio.write_image(fig,"./plot/graph_" + str(iter_number) + ".png")
        print("Done!")  
         
        