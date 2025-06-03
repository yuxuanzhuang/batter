# Copy from Amber FETools
import os
import numpy as np
from loguru import logger

class RemdLog:
    def __init__(self, inputfile):
        if not os.path.isfile(inputfile):
            raise FileNotFoundError(f"Input file '{inputfile}' does not exist.")
        self.inputfile = inputfile
        self.replica_trajectory = None
        self.replica_state_count = None
        self.replica_ex_count = None
        self.replica_ex_succ = None
        self.ARs = None

    def read_log(self):
        (self.replica_trajectory, self.replica_state_count,
         self.replica_ex_count, self.replica_ex_succ, self.ARs) = read_rem_log(self.inputfile)

    def analyze(self):
        if self.replica_trajectory is None:
            raise ValueError("Replica trajectory is not initialized. Call read_log() first.")
        return remd_analysis(self.replica_trajectory, self.ARs)
    
def read_rem_log(inputfile):
    print("")
    print("Analyzing remlog file", )
    print("")

    np.set_printoptions(precision=2, linewidth=150,
                        formatter={'int': '{:2d}'.format})

    rep=[]
    neigh=[]
    succ=[]
    count=0
    n_replica=0
    try:
        with open(inputfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise Exception(f"File '{inputfile}' not found.")

    for line in lines:
        count=count+1
        if line[0] != '#':
            rep.append(int(line[0:6]))
            neigh.append(int(line[6:12]))
            if line[66] == 'T' or line[66] == 'F':
                succ.append(line[66:67])
            else:
                succ.append(line[91:92])
        if count>200:n_replica=max(rep[0:200])
    
    f.close()
    print("Done reading the remlog")



    n_replica=max(rep[0:200])  
    n_step=int(len(rep)/n_replica)
    print("# of Replicas:", n_replica, "# of Steps:", n_step)
    n_state=n_replica
    ARs = [line.strip().split()[-1] for line in lines[-n_replica:-1]]
    ARs = [float(AR) for AR in ARs]
    

    replica_trajectory=np.zeros((n_replica, n_step+1), np.int64)
    replica_state_count=np.zeros((n_replica, n_state), np.int64)
    replica_ex_count=np.zeros((n_replica, n_state-1), np.int64)
    replica_ex_succ=np.zeros((n_replica, n_state-1), np.int64)

    for i in range(n_replica):
        replica_trajectory[i][0]=i+1
        replica_state_count[i][i]=1

    for m in range(n_step):
        replica_trajectory[0:n_replica, m+1]=replica_trajectory[0:n_replica, m]
        for i in range((m+1)%2,n_replica-1,2): 
            k=m*n_replica+i
            x=np.where(replica_trajectory[:,m+1]==i+1)
            y=np.where(replica_trajectory[:,m+1]==i+2)
            replica_ex_count[x[0],i]+=1
            if succ[k]=='T':
                replica_ex_succ[x[0],i]+=1
                replica_trajectory[y[0],m+1]=i+1
                replica_trajectory[x[0],m+1]=i+2

        for j in range(n_replica) :
            replica_state_count[j, replica_trajectory[j,m+1]-1]+=1
    
    return replica_trajectory, replica_state_count, \
        replica_ex_count, replica_ex_succ, ARs


def remd_analysis(replica_trajectory, ARs):

    n_replica=np.size(replica_trajectory, 0)
    n_state=n_replica
    n_step=np.size(replica_trajectory, 1)

    print("Analyzing", n_replica, n_step)

    h1n=[]
    hn1=[] 
    k1n=[]
    kn1=[] 
    trip_count_1n=[0]*n_replica
    trip_count_n1=[0]*n_replica
    for i in range(n_replica):
        first_step_at_1=-1
        first_step_at_n=-1
        last_step_at_1=-1
        last_step_at_n=-1
        at_1=0
        at_n=0
      
        for j in range(n_step):
            if replica_trajectory[i][j] == 1:
                last_step_at_1=j
                if at_1 ==0:
                    at_1=1
                    at_n=0
                    first_step_at_1=j
                if first_step_at_n >=0:
                    #print("Rep #",i, 'At state 1:', first_step_at_n, j, j-first_step_at_n);
                    hn1.append(j-first_step_at_n)
                    first_step_at_n=-1 
                    trip_count_n1[i]+=1
                if last_step_at_n >=0:
                    #print('**At state 1:', last_step_at_n, j, j-last_step_at_n);
                    kn1.append(j-last_step_at_n)
                    last_step_at_n=-1 
            if replica_trajectory[i][j] == n_replica:
                last_step_at_n=j
                if at_n ==0:
                    at_n=1
                    at_1=0
                    first_step_at_n=j
                    if first_step_at_1 >=0:
                        #print("Rep #",i, 'At state N:', first_step_at_1, j, j-first_step_at_1);
                        h1n.append(j-first_step_at_1)
                        first_step_at_1=-1 
                        trip_count_1n[i]+=1
                    if last_step_at_1 >=0:
                        #print('**At state N:', last_step_at_1, j, j-last_step_at_1);
                        k1n.append(j-last_step_at_1)
                        last_step_at_1=-1 


    output_data = {}
    if len(h1n)==0 or len(hn1)==0:
        print("")
        print("No single pass found", )
        print("")
        output_data["Average single pass steps:"] = 1.e+8
        output_data["Round trips per replica:"] = 0.
        output_data["Total round trips:"] = 0.
        output_data["neighbor_acceptance_ratio"] = ARs
    else:
        hh=h1n+hn1
        mean_value = np.mean(hh)
        output_data["Average single pass steps:"] = float(mean_value)
        output_data["Round trips per replica:"] = float(len(hh)/2/n_replica)
        output_data["Total round trips:"] = float(len(hh)/2)
        output_data["neighbor_acceptance_ratio"] = ARs

    return output_data


def get_remd_info(inputfile):
    reptraj, nstate, nexch, nsucc, ARs = read_rem_log(inputfile)
    #np.set_printoptions(precision=5, linewidth=150,
    #                    formatter={'int': '{:2d}'.format})
    return remd_analysis(reptraj, ARs)