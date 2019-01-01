import os
import numpy as np
from functools import reduce


def prod(a,b):
    DICTS = []
    newvals = b[1]
    newkeys = b[0]
    for d in a:
        for i in np.arange(len(newvals)): 
            K = list(d.keys()) + [newkeys]
            V = list(d.values()) + [newvals[i]] 
            newDict = {}
            for k,v in zip(K,V):
                newDict[k] = v
            DICTS += [newDict]
    return(DICTS)

args = dict()
args["dataset"] = ["box"]
args["n1"] = [10]
args["n2"] = [10]
args["sigma_i"] = [3]
args["sigma_f"] = [0.1]
args["eps_i"] = [1]
args["eps_f"] = [1]

args["ntype"] = ["GAUSSIAN"]
args["plotmode"] = ["CLASS_COLOR"]
args["gridmode"] = ["RECTANGLE"]
args["initmode"] = ["PCAINIT"]
args["n_iter"] = [10000]
args["plot_iter"] = [100]
args["run_id"] = np.arange(1)
def allComb(args):
    return(reduce(lambda a,b: prod(a,b), list(zip(args.keys(),args.values())), [{}]))

print(allComb(args))
print(len(allComb(args)))

def run_program(args):
    #Run program with set learning rate
    comm1  = "python3 main.py " 
    for key in args.keys():
        comm1 = comm1 + " --" + key
        if args[key] != None:
            comm1 = comm1 + "=" + str(args[key])
            
    print(comm1)
    os.system(comm1)

for x in allComb(args):
    run_program(x)

'''
P = list(np.round(np.arange(0.00020,0.001,0.00010),4 ))

for x in zip(P,P):
    args["learning_rate_G"] = x[0]
    args["learning_rate_D"] = x[1]
    
    #Clean chckpoint dir, logs dir
    os.system("rm ./checkpoint/club_64_64_64/*")
    os.system("rm ./logs/*")
    

    
    #Get corresponding output path, create dir
    OUT_PATH  = "./samples" + "/D="  + str(args["learning_rate_D"]) + \
                 "_G="  + str(args["learning_rate_G"])
    
    if not os.path.exists(OUT_PATH): 
        os.mkdir(OUT_PATH)

    comm2 = "cp ./samples/* " + OUT_PATH 
    
    
    
    print(comm2)    
    os.system(comm2)
'''    