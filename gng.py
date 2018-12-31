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

plotly.tools.set_credentials_file(username='isonettv', api_key='2Lg1USMkZAHONqo82eMG')

Mode = Enum("Mode","TRAIN TEST PRED")



class gng(object):


    

    ''' Initializes the Growing Neural Gas Network
        @param sess: the current session
        @param args: object containing arguments (see main.py)
    ''' 
    def __init__(self, sess, args): # Parameters
        
        self.sess = sess
        
        #TODO: use arguments to get parameters 
        self.lamb = 100
        self.eps_b = 0.1
        self.eps_n = 0.01
        self.alpha = 0.5
        self.beta = 0.0005
        self.amax = 40
        self.m = 300
        self.n_iter = 40000
        self.plot_iter = 600
        
        
         #Get datasets
        if args.dataset == "isolet":
            temp = gd.get_isolet(test_subset = args.cv)
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
            self.ds = temp["train"]
        else:
            self.ds = temp
        
        #Normalize Dataset and store df for PCA
        self.ds_size = self.getNumElementsOfDataset(self.ds)
        self.ds_inputdim = self.getInputShapeOfDataset(self.ds)
        print(self.ds_inputdim)
        self.ds = self.normalizedDataset(self.ds)

        
        self.ds = temp["dataset"].shuffle(buffer_size=10000).repeat()
        
        #Get PCA for dataset
        print("Generating PCA for further plotting...")
        self.pca = PCA(n_components=3)
        self.input_pca = self.pca.fit_transform(temp["df_x"])
        self.input_pca_scatter = self.inputScatter3D()
        print("Done!")
        print(temp["df_x"])
        
        if args.dataset == "box":
            self.ds = gd.get_box()
        
        #Now generate iterators for dataset
        self.iterator_ds = self.ds.make_initializable_iterator()
        self.iter_next = self.iterator_ds.get_next()
        
        
        
        # tf Graph input
        self.X_placeholder = tf.placeholder("float", [self.ds_inputdim])
        self.W_placeholder = tf.placeholder("float", [None,self.ds_inputdim])
        
        print("Initializing graph and global vars...")
        self.init_graph()
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        print("Done!")
        
       
        
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
        return d
    
    
    
    
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
        return({"df_x":  df_x,
                "df_y":  df_y,
                "dataset": tf.data.Dataset.from_tensor_slices({"X":df_x,"Y":df_y}) \
                })
        
        
       
    
    ''' Initializes the GNG graph
    '''
    def init_graph(self):
        #initial topology
        self.g = ig.Graph()
        self.g.add_vertices(2)
        self.g.add_edges([(0,1)])
       
        #set ages and accumulated error to 0
        edge = self.g.es.find(_between=((0,), (1,)) )
        edge["age"] = 0
        self.g.vs["accumulated_error"] = 0        
        
        #Set first 2 values
        self.sess.run(self.iterator_ds.initializer)
        c1 = self.sess.run(self.iter_next)["X"]
        c2 = self.sess.run(self.iter_next)["X"]
        self.W =  np.asarray([c1,c2],dtype=np.float32)
        
        
        
    def getNeighborhood(self,i):
        return [0,1]
    
    def build_model(self):
        X = self.X_placeholder
        W = self.W_placeholder
        #Step 3: Calculate two nearest
        self.dist_vecs = tf.map_fn(lambda w: X - w,W)
        self.squared_distances = tf.map_fn(lambda w: 2.0*tf.nn.l2_loss(X - w),W)
        self.s = tf.math.top_k(-self.squared_distances,k=2) 
        self.s1_2d = self.s.indices[0]
        self.s2_2d = self.s.indices[1]

    def get_random_choice(self):
        return (tf.random.uniform(shape=[],minval=0,maxval=(self.n_input-1),dtype=tf.int64))
    
    def train(self):
        #Run initializer
        self.sess.run(self.init)
        self.sess.run(self.iterator_ds.initializer)
        
        
        for current_iter in np.arange(self.n_iter):
            added_edge = False
            print("Iteration {}".format(current_iter))
            nxt = self.sess.run(self.iter_next)["X"]
        #Step 3
            s1, s2, d, sq = self.sess.run([self.s1_2d, self.s2_2d, self.dist_vecs,\
                                            self.squared_distances],\
                                           feed_dict = {self.X_placeholder: nxt,\
                                                        self.W_placeholder: self.W})
        #Step 4a: If a connection between s1_2d and s2_2d does not exist already, create it
            if not self.g.are_connected(s1, s2):
                self.g.add_edges([(s1,s2)])

                added_edge = True
        #Step 4b:  Refresh edge
            self.g.es.find(_between=((s1,), (s2,)) )["age"] = 0
            
        #Step 5: Add squared error of winner to its accumulated error
            self.g.vs[s1]["accumulated_error"] += sq[s1]
            
            #Get neighborhood of s1_2d
            N_s1 = self.g.neighbors(s1)
            
        #BEGIN STEP 6
            #Step 6a : Modify winner
            self.W[s1,] += self.eps_b * d[s1]
            #Step 6b : Modify winner's neighbors
            for nbor in N_s1:
                self.W[nbor,] += self.eps_n * d[nbor]
        #END STEP 6
            
        #BEGIN Step 7,8
            vertices_to_delete = []
            #Increment age of all edges emanating from winner, removing those greater than amax
            for nbor in N_s1:
                age = self.g.es.find(_between=((s1,), (nbor,)) )["age"]
                edge = self.g.es.find(_between=((s1,), (nbor,)) )
                edge["age"] = age + 1
                if age + 1 > self.amax:
                    #Remove edge
                    #print("Removing edge...")
                    self.g.delete_edges(edge)
                    #Check if neighbor still has neighbors
                    if len(self.g.neighbors(nbor)) == 0:
                        #print("Removing Vertex...")
                        vertices_to_delete.append(nbor)
                        
            if len(self.g.neighbors(s1)) == 0:
                vertices_to_delete.append(nbor)
            #Remove vertices
            self.g.delete_vertices(vertices_to_delete)
            self.W = np.delete(self.W,vertices_to_delete,0)
        #BEGIN Step 7,8
            
            
        #Step 9: IF  iter is multiple of lambda, THEN Add vertex
            if self.g.vcount() <= self.m and current_iter > 0\
              and current_iter % self.lamb == 0:
                added_edge = True
                #Determine unit with maximum accumulated error
                q = np.argmax(self.g.vs["accumulated_error"])
                #Determine q's neighbor with max acc error
                f = self.g.neighbors(q)[\
                            np.argmax(self.g.vs[self.g.neighbors(q)]["accumulated_error"])\
                            ]
                
                #Decrease q and f acc error
                self.g.vs[q]["accumulated_error"] += (-self.alpha)*self.g.vs[q]["accumulated_error"]
                self.g.vs[f]["accumulated_error"] += (-self.alpha)*self.g.vs[f]["accumulated_error"]
                
                #Remove edge between q,f
                edge_qf = self.g.es.find(_between=((q,), (f,)) )
                self.g.delete_edges(edge_qf)
                
                
                #Add new unit
                self.g.add_vertices(1)
                r = self.g.vcount()-1
                w = 0.5* (self.W[q,:] + self.W[f,:])
                w_err = 0.5* (self.g.vs[q]["accumulated_error"]+\
                              self.g.vs[f]["accumulated_error"])
                self.W = np.concatenate((self.W,[w]))
                
                #Add its error
                self.g.vs[r]["accumulated_error"] = w_err 
                
                #Add edges from new unit to q and f
                self.g.add_edges([(r,q),(r,f)])
                #set age of these edges  to zero
                edge_rq = self.g.es.find(_between=((r,), (q,)) )
                edge_rf = self.g.es.find(_between=((r,), (f,))) 
                edge_rq["age"] = 0
                edge_rf["age"] = 0

        #Step 10: Decrease acc error of all reference vectors
            self.g.vs["accumulated_error"] = \
                [ err + err*(-self.beta) for err in self.g.vs["accumulated_error"]]
                 
            if  current_iter % self.plot_iter == 0:
                self.prettygraph(current_iter,s1,s2)
         
        self.prettygraph(self.n_iter,s1=None,s2=None,online=True)
        
        
    def inputScatter3D(self):
        Xn = self.input_pca[:,0]
        Yn = self.input_pca[:,1]
        Zn = self.input_pca[:,2]
            
        trace0=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='input',
                       marker=dict(symbol='circle',
                                     size=1,
                                     color="green",
                                     line=dict(color='rgb(50,50,50)', width=0.25)
                                     ),
                       text="",
                       hoverinfo='text'
                       )
        return(trace0)    
    def prettygraph(self,iter_number, s1,s2,online = False):
        
        trace0 = self.input_pca_scatter
        
        W = self.pca.transform(self.W)
        Xn=W[:,0]# x-coordinates of nodes
        Yn=W[:,1] # y-coordinates
        Zn=W[:,2] # z-coordinates
        
        edge_colors = []
        Xe=[]
        Ye=[]
        Ze=[]
        
        num_pallete = 1000
        for e in self.g.get_edgelist():
            col  = self.g.es.find(_between=((e[0],), (e[1],)),)["age"]
            col = float(col)/float(self.amax)
            col = min(num_pallete-1, int(num_pallete  * col))
            edge_colors += [col,col]
            
            Xe+=[W[e[0],0],W[e[1],0],None]# x-coordinates of edge ends
            Ye+=[W[e[0],1],W[e[1],1],None]# y-coordinates of edge ends
            Ze+=[W[e[0],2],W[e[1],2],None]# z-coordinates of edge ends
           
        
        #Create Scaling for edges based on Age
        
        pal_V = ig.GradientPalette("blue", "red", num_pallete)
        pal_E = ig.GradientPalette("black", "white", num_pallete)
        
        v_order =  np.argsort(self.g.vs["accumulated_error"])
        v_len = len(v_order)
        v_colors = np.zeros(shape=[v_len],dtype = np.int32)
        for i in np.arange(v_len):
            v_colors[v_order[i]] = int(np.round(i*float(num_pallete-1) /float(v_len-1)) ) 
        
        #TODO: Create colors for vertices (s1_2d, s2_2d, neighbor of s1_2d, none)
        v_colors = [np.asscalar(a) for a in v_colors]
        
        
        edge_colors = pal_E.get_many(edge_colors)
        v_colors = pal_V.get_many(v_colors)
        
        if s1 != None:
            v_colors[s1] = "yellow"
        if s2 != None:
            v_colors[s2] = "yellow"
        


         
        trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=dict(color=edge_colors,
                                  width=3),
                       hoverinfo='none'
                       )
        
        reference_vec_text = ["m" + comm1(x) for x in np.arange(self.W.shape[0])]
        trace2=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='reference_vectors',
                       marker=dict(symbol='circle',
                                     size=6,
                                     color=v_colors,
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=reference_vec_text,
                       hoverinfo='text'
                       )
        
        axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )
        
        layout = go.Layout(
                 title="Visualization of GNG",
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
            py.iplot(fig)
        pio.write_image(fig,"./plot/graph_" + comm1(iter_number) + ".png")
        print("Done!")  
         
        