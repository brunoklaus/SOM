import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

import get_datasets as gd
import re
from pandas.core.frame import DataFrame
from enum import Enum

Mode = Enum("Mode","TRAIN TEST PRED")


class nn(object):

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        
        
        # Parameters
        self.learning_rate = args.lr
        self.training_epochs = args.epoch
        self.batch_size = args.batch_size
        self.display_step = 1
        
        
        
        # Network Parameters
        self.stddev = 1
        self.lamb = args.lamb
        self.dropout = args.dropout
        self.momentum = args.momentum
        self.dataset_str = args.dataset
        self.cv = args.cv
        
        print("MOMENTUM: " + comm1(args.momentum) )
        
        if self.dataset_str == "iris":
                self.training_epochs = 200
                self.batch_size = 15
        elif self.dataset_str == "wine":
                self.training_epochs = 10000
                self.batch_size = 498
        else:
                self.training_epochs = 350
                self.batch_size = 700
	
        
        self.n_input = gd.get_dataset_metadata(args.dataset)["num_input"] #Size of data input
        self.n_classes = gd.get_dataset_metadata(args.dataset)["num_classes"] # total classes
        
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("int32", [None,self.n_classes])
        
        #Get datasets
        if args.dataset == "isolet":
            temp = gd.get_isolet(test_subset = args.cv)
        elif args.dataset == "iris":
            temp = gd.get_iris(args.cv)
        elif args.dataset == "wine":
            temp = gd.get_wine(args.cv)
        else:
            raise ValueError("Bad dataset name")
        
        self.train_dataset = temp["train"]
        self.test_dataset = temp["test"]
        
        #Get dataset sizes
        self.total_examples_train = self.getNumElementsOfDataset(self.train_dataset)
        self.total_examples_test = self.getNumElementsOfDataset(self.test_dataset)
        print("Total Number of examples in train_dataset : {:}".format(self.total_examples_train))
        print("Total Number of examples in test_dataset : {:}".format(self.total_examples_test))
        
        
        
        self.weights = []
        self.biases = []
        self.activations = []
        
        self.arch_str = args.arch
        if(self.arch_str[len(self.arch_str)-1] == ";"):
            self.arch_str = self.arch_str[:-1]
        
        i = 0
        num_neurons = 0
        for x in self.arch_str.split(";"):
            i += 1
            print(x)
            if(None == re.search("h[0-9]+(r|s)", x)):
                raise ValueError("Bad architecure string")
            else:
                #Update num of neurons in last layer
                if i == 1:
                    last_numneurons = self.n_input
                else:    
                    last_numneurons = num_neurons
                    

                p = re.compile("h([0-9]+)([r|s])")
                m = p.match(x)
                num_neurons = int(m.group(1))
                act = m.group(2)
                print("Hidden Layer {}: neurons={};act={}".format(i,num_neurons,act))
                self.weights.append(
                    tf.Variable(tf.random_normal([last_numneurons, num_neurons],\
                                                 stddev=2./np.sqrt(last_numneurons+num_neurons)),\
                                                 name="weight_hidden{}".format(i))
                    )
               
                self.biases.append(
                    tf.Variable(tf.random_normal([num_neurons],\
                                                 stddev=2./np.sqrt(last_numneurons+num_neurons)),\
                                                 name="bias_hidden{}".format(i))
                    )
                self.activations.append(act)
                
        #Now add output layer
        self.weights.append(
            tf.Variable(tf.random_normal([num_neurons, self.n_classes],\
                                            stddev=2./np.sqrt(self.n_classes+num_neurons)),\
                                         name="weight_out")
            )
        self.biases.append(
            tf.Variable(tf.random_normal([self.n_classes],\
                                            stddev=2./np.sqrt(self.n_classes+num_neurons)),\
                                         name="bias_out")
            )
        print(self.weights)
        print(self.biases)
        
        '''
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1],name="weight_hidden1")),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2],name="weight_hidden2")),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes],name="weight_out"))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]),name="bias_hidden1"),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]),name="bias_hidden2"),
            'out': tf.Variable(tf.random_normal([self.n_classes]),name="bias_out")
        }
        '''
        
        
    def multilayer_perceptron(self):
        for i in np.arange(len(self.weights)-1):
            print(i)
            if i == 0:
                layer = tf.add(tf.matmul(self.X, self.weights[i]), self.biases[i])
            else:
                layer = tf.add(tf.matmul(layer, self.weights[i]), self.biases[i])
            if (self.activations[i] == "r"):
                layer = tf.nn.relu(layer, name="layer_{}".format(i+1))
            elif (self.activations[i] == "s"):
                layer = tf.nn.sigmoid(layer, name="layer_{}".format(i+1))
            else:
                raise ValueError("Unknown activation")
            if self.dropout > 0:
                layer = tf.nn.dropout(layer, self.dropout)
            
        self.logits = ( (\
                         tf.matmul(layer, self.weights[len(self.weights)-1]) + self.biases[len(self.weights)-1]))
        out_layer = tf.nn.softmax(self.logits,axis=-1)

         
        '''  # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1']))
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        # Output fully connected layer with a neuron for each class
        out_layer = (tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
        
        #pred = tf.nn.softmax(out_layer) # Apply softmax to logits
        '''
        return out_layer
        
    def build_model(self):
        self.counter = tf.get_variable(name="counter",shape=[],dtype=tf.int32)

        # Construct model
        self.pred = self.multilayer_perceptron()
        
       
        
        #print(self.Y)
        
        use_log_loss = False
        
        if use_log_loss:
            # Use MSE
            self.error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,logits=self.logits)
        else:
            # Use MSE
            self.error = 0.5*tf.square(tf.cast(self.Y,dtype=tf.float32) - self.pred)
            
       
        
        #Add regularization
        if self.lamb > 0:
            reg = tf.reduce_sum(tf.square(self.weights[0]))
            for i in np.arange(start=1,stop=len(self.weights)):
                reg = reg +  tf.reduce_sum(tf.square(self.weights[i]))
            self.batch_errors =  tf.reduce_mean(self.error + self.lamb * reg,axis=-1,name="mse_and_reg")
        else:
            #MSE only
            self.batch_errors =  tf.reduce_mean(self.error,axis=-1,name="mse")
        
        #SUm over batches
        self.loss_op =  tf.reduce_mean(self.batch_errors)
        
        
        
        

        if self.momentum == True:
            self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
        self.train_op = self.optimizer.minimize(self.loss_op)
        
        
        self.actual_pred = tf.cast(tf.argmax(self.pred,axis=-1),dtype=tf.int32)
        self.true_pred = tf.cast(tf.argmax(self.Y,axis=-1),dtype=tf.int32)
        
        
        #Extra info for number of misclassifications
        self.num_misc = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(\
                           self.actual_pred,self.true_pred)), tf.float32))
        
        self.temp1 = self.num_misc
        
        
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        t_vars = tf.trainable_variables()
        for var in t_vars: print(var.name)
        
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
    
    def get_self_string(self):
        return("arch={}_lr={}_momentum={}_lamb={}_dropout={}_dataset={}_cv={}").\
            format(self.arch_str,self.learning_rate,self.momentum,self.lamb,self.dropout,\
                   self.dataset_str,self.cv)
    
    def cycle(self,batched_dataset,iterator,next_element,mode, return_pred=False):
        
        trues = []
        actuals = []
        self.sess.run(iterator.initializer)

        total_cost = 0.
        total_batch = 0
        
        total_misc = 0
        while True:
            try:
                total_batch += 1
                
                res = self.sess.run(next_element)
                if mode == Mode.TRAIN:
                    _, c, misc  = self.sess.run([self.train_op, self.loss_op, self.num_misc], feed_dict={self.X: res["X"],
                                                                self.Y: res["Y"]})
                    
                    # Compute average loss
                    total_cost += c
                    total_misc += misc
                elif mode == Mode.TEST:
                    c, misc = self.sess.run([self.loss_op,self.num_misc], feed_dict={self.X: res["X"],
                                                                    self.Y: res["Y"]})
                    # Compute average loss
                    total_cost += c
                    total_misc += misc
                elif mode == Mode.PRED:
                    
                    t, a = self.sess.run([self.true_pred,self.actual_pred], feed_dict={self.X: res["X"],
                                                                    self.Y: res["Y"]})
                    trues.extend(t)
                    actuals.extend(a)
                else:
                    raise ValueError("Bad mode")    
                
                
                
                
            except tf.errors.OutOfRangeError:
                break
        avg_cost = total_cost / total_batch
        if mode == Mode.TRAIN:
            avg_misc = total_misc / self.total_examples_train
        else:
            avg_misc = total_misc / self.total_examples_test
        if mode == mode.PRED:
            #print(actuals)
            return (trues,actuals)
        else: 
            return (avg_cost,avg_misc)
            
            
        
    def train(self):
       
        
        df_cols = ["train_avg_loss","train_error_rate","test_avg_loss","test_error_rate"]
        df_index = np.arange(self.training_epochs)
        df = DataFrame(index=df_index,columns=df_cols)
        
        self.sess.run(self.init)
        
        batched_train =  self.train_dataset.shuffle(buffer_size=10000).\
                                                    batch(self.batch_size)
        batched_test =  self.test_dataset.shuffle(buffer_size=10000).\
                                                    batch(self.batch_size)
                                                    
        iterator_train = batched_train.make_initializable_iterator()
        train_next = iterator_train.get_next()
        iterator_test = batched_test.make_initializable_iterator()
        test_next = iterator_test.get_next()
                                                    
        

        
        for epoch in range(self.training_epochs):
            #Training Cycle
            avg_cost_train,avg_misc_train = self.cycle(batched_train,iterator_train,train_next,Mode.TRAIN)
            #Test Cycle
            avg_cost_test, avg_misc_test = self.cycle(batched_test,iterator_test,test_next, Mode.TEST)
            
            #Write acc results to df
            df.iloc[epoch,:] = [avg_cost_train,avg_misc_train,avg_cost_test,avg_misc_test]
            
            
            
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),\
                       "train_cost={:.5f};train_misc={:.5f};test_cost={:.5f};test_misc={:.5f}".\
                       format(avg_cost_train,avg_misc_train,avg_cost_test,avg_misc_test))
        print("Optimization Finished!")
        
        #Save training data to CSV
        df.to_csv("./results_2/" + self.dataset_str + "/acc/" + self.get_self_string() + ".csv") 
        
        #Now save final  train predictions
        df_trainpred_cols = ["true_pred","actual_pred"]
        df_trainpred_index = np.arange(self.total_examples_train)
        df_trainpred = DataFrame(index=df_trainpred_index,columns=df_trainpred_cols)
        t, a = self.cycle(batched_train,iterator_train,train_next,Mode.PRED)
        df_trainpred.iloc[:,0] = t
        df_trainpred.iloc[:,1] = a
        df_trainpred.to_csv("./results_2/" + self.dataset_str + "/trainpred/" + self.get_self_string() + ".csv") 
        #..And test predictions
        df_testpred_cols = ["true_pred","actual_pred"]
        df_testpred_index = np.arange(self.total_examples_test)
        df_testpred = DataFrame(index=df_testpred_index,columns=df_testpred_cols)
        t, a = self.cycle(batched_test,iterator_test,test_next,Mode.PRED)
        df_testpred.iloc[:,0] = t
        df_testpred.iloc[:,1] = a
        df_testpred.to_csv("./results_2/" + self.dataset_str + "/testpred/" + self.get_self_string() + ".csv") 
              
        
