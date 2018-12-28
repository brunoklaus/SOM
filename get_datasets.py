import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, KFold
            

#print(np.shape(np.where(df.iloc[:,df.shape[1] - 1] == 1 )) )
#print(df.iloc[np.where(df.iloc[:,df.shape[1] - 1] == 1 )])


def get_dataset_metadata(s):
    if s == "isolet":
        return{"num_classes":26, "num_input": 617} 
    elif s == "wine":
        return{"num_classes":7, "num_input": 11} 
    elif s == "iris":
        return{"num_classes":3, "num_input": 4} 
    else:
        raise LookupError("Did not find dataset metadata")
    

def getFeatures(df):
    return(df.iloc[:,0:(df.shape[1] - 1)])
def getClass(df):
    return(df.iloc[:,df.shape[1] - 1])    
        
def get_isolet(test_subset = 5):
    #Read  isolet 1 through 4 
    df = pd.read_csv('./isolet/isolet1+2+3+4_normalized.data', header=0)
    #Read  isolet 5 
    iso5 = pd.read_csv('./isolet/isolet5_normalized.data', header=0)
    

    
    # Get location of all utterances of letter A (i.e. 1st of alphabet)
    loc1= np.reshape(np.where(df.iloc[:,df.shape[1] - 1] == 1 ), [-1])

    i = 0
    #Will Hold each of the 5 isolet subsets
    isolet = [None] * 5 
    for loc1_begin in np.arange(0,loc1.size,60):
      
        if loc1_begin + 60 >= loc1.size:
            isolet[i] = (df.iloc[loc1[loc1_begin]:,:])       
        else:
            isolet[i] = (df.iloc[ loc1[loc1_begin]:(loc1[loc1_begin + 60]),:]) 
        i += 1        
    isolet[4] = iso5
      
    train_isolet = pd.concat(list( map(lambda j: isolet[j-1],\
                            filter(lambda i: i != test_subset, range(1,6))) ))
    test_isolet = pd.concat(list( map(lambda j: isolet[j-1],\
                            filter(lambda i: i == test_subset, range(1,6))) ))
    
    assert(train_isolet.shape[0] + test_isolet.shape[0] == df.shape[0] + iso5.shape[0])
    assert(train_isolet.shape[1] == getFeatures(train_isolet).shape[1] + 1)
    
    print(iso5.shape)
    
    #return({"train_x":getFeatures(train_isolet), "train_y":getClass(train_isolet),\
    #        "test_x":getFeatures(test_isolet), "test_y":getClass(test_isolet),\
     #       })
    
     
    return ({"train":tf.data.Dataset.from_tensor_slices({"X":getFeatures(train_isolet),\
                 "Y":tf.one_hot((-1) + tf.cast(getClass(train_isolet),dtype=tf.int32),26) }),\
            "test":tf.data.Dataset.from_tensor_slices({"X":getFeatures(test_isolet),\
                 "Y":tf.one_hot((-1) + tf.cast(getClass(test_isolet),dtype=tf.int32),26) })\
})

def get_iris(cv = 1):
    #Read  isolet 1 through 4 
    df = pd.read_csv('./iris/iris_normalized.data')
    X = getFeatures(df)
    Y = getClass(df)
    skf = StratifiedKFold(n_splits=5,random_state = 19)
    
    i = 0
    for train_index, test_index in skf.split(X, Y):
        i += 1
        if (i == cv):
            train_x = X.iloc[train_index,:]
            train_y = Y.iloc[train_index]
            test_x = X.iloc[test_index,:]
            test_y = Y.iloc[test_index]
            print(test_y.shape)
            return({"train":tf.data.Dataset.from_tensor_slices({"X":train_x,\
                            "Y":tf.one_hot((-1) + tf.cast(train_y,dtype=tf.int32),\
                                           get_dataset_metadata("iris")["num_classes"]) }),\
                    "test":tf.data.Dataset.from_tensor_slices({"X":test_x,\
                            "Y":tf.one_hot((-1) + tf.cast(test_y,dtype=tf.int32),\
                                           get_dataset_metadata("iris")["num_classes"]) })\
                    })                                          
            
    return None

def get_grid(size=10):
    X =  pd.DataFrame(0,index=np.arange(size*size),columns=np.arange(3))
    Y =  pd.DataFrame(0,index=np.arange(size*size),columns=np.arange(1))
    k = 0
    for i in np.arange(size):
        for j in np.arange(size):
            X.iloc[k,:] = [i, j, 0]
            k+=1   
            
    return tf.data.Dataset.from_tensor_slices({"X":X,"Y":Y})
def get_box():
    X = [0]
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.map(lambda x: {"X":[tf.random.uniform([], -1.57, 1.57),\
                                          tf.random.uniform([], -1.57, 1.57),0] }).repeat()        
    return dataset


def get_wine(cv = 1):
    #Read  isolet 1 through 4 
    df = pd.read_csv('./wine/winequality-white_normalized.csv',sep=",")
    X = getFeatures(df)
    Y = getClass(df)
    
    skf = KFold(n_splits=10,random_state = 19)
   
    i = 0
    for train_index, test_index in skf.split(X, Y):
        i += 1
        if (i == cv):
            train_x = X.iloc[train_index,:]
            train_y = Y.iloc[train_index]
            test_x = X.iloc[test_index,:]
            test_y = Y.iloc[test_index]
           
            num_classes = get_dataset_metadata("wine")["num_classes"]
            
            return({"train":tf.data.Dataset.from_tensor_slices({"X":train_x,\
                            "Y":tf.one_hot((-3) + tf.cast(train_y,dtype=tf.int32),\
                                           num_classes) }),\
                    "test":tf.data.Dataset.from_tensor_slices({"X":test_x,\
                            "Y":tf.one_hot((-3) + tf.cast(test_y,dtype=tf.int32),\
                                           num_classes) })\
                    })                                          
            
    return None



get_iris(1)
        