#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_npy_file>"
    exit 1
fi

# Extract the provided argument (path to npy file)
npy_file="$1"

echo $1

# Check if the provided file exists
if [ ! -f "$npy_file" ]; then
    echo "Error: File not found: $npy_file"
    exit 1
fi

# Perform the necessary computations using Python
python_script=$(cat <<END

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import precision_recall_fscore_support
from prettytable import PrettyTable
import warnings
import pandas as pd

warnings.filterwarnings("ignore")




data = np.load('$npy_file',allow_pickle=True)

df = pd.DataFrame(data=data)
df[0] = np.arange(0,data.shape[0])

for i in range(data[:,2].shape[0]):
    data[:,2][i] = data[:,2][i].flatten()
    
for i in range(data[:,1].shape[0]):
    data[:,1][i] = data[:,1][i].flatten()
 
data = data[:,:4]

train = int(0.8*data.shape[0])
val = int(0.2*data.shape[0])
test = int(0*data.shape[0])

train_data = data[:train]
val_data = data[train:val+train]
test_data = data[train+val:train+val+test]

class Distance():
    def __init__(self,k,dm):
        self.distance_metric_set = dm
        pass

    def Euclidean(self,v1,v2):
        return np.sqrt(np.sum((v1-v2)**2,axis=1))
    
    def Manhatten(self,v1,v2):
        return np.sum(np.abs(v1-v2),axis=1)
    
    def Cosine(self,v1,v2):
        return 1 - (np.dot(v1,v2)/(np.linalg.norm(v1,axis=1)*\
            np.linalg.norm(v2)))

mapper_lbl = {k:v for k,v in enumerate(list(np.unique(data[:,-1].flatten())),start=1)}
mapper_lbl_rev = {v:k for k,v in enumerate(list(np.unique(data[:,-1].flatten())),start=1)}


class KNNModel(Distance):
    
    mp_lbl = mapper_lbl
    mp_lbl_rev = mapper_lbl_rev
    
    def __init__(self,k,dm,enc_type,data):
        self.k = k
        self.data = data
        self.samples = data.shape[0]
        self.encoder_type=enc_type
        self.distance_metric = self.distance_metric_finder(dm)
        self.mapper_str_to_int_vectorize = np.vectorize(self.mapper_str_to_int)
                
    def distance_metric_finder(self,dm):
        if dm.lower() == "manhatten":
            return self.Manhatten
        if dm.lower() == "euclidean":
            return self.Euclidean
        if dm.lower() == "cosine":
            return self.Cosine
    
    def mapper_str_to_int(self,label):
        return KNNModel.mp_lbl_rev[label]


    def inference(self,test_samples):
        return_list = []
        
        collection_of_vectors = np.stack(self.data[:,self.encoder_type])
        for test_smpl in test_samples[:,self.encoder_type]:
            
            distances = self.distance_metric(collection_of_vectors,test_smpl)
            labels = self.mapper_str_to_int_vectorize(self.data[:,3])
            
            top_k = np.column_stack((distances, labels))
            
            top_k = top_k[np.argsort(distances)]
            top_k = top_k[:self.k]
            top_k = np.array(top_k)

            unique, counts = np.unique(top_k[:,1], return_counts=True)
            max_freq_count = np.max(counts)
            max_freq_labels = unique[counts == max_freq_count]
            resultant_shortlisted_neighbors = top_k[np.isin(top_k[:, 1], max_freq_labels)]
            return_list.append(resultant_shortlisted_neighbors[np.argmin(resultant_shortlisted_neighbors[:,0])][1])
            
        return return_list
    
    def validation_metrics_table(self,y_true,y_pred):
        a_macro = precision_recall_fscore_support(y_true,y_pred,average='macro')
        a_micro = precision_recall_fscore_support(y_true,y_pred,average='micro')
        a_weighted = precision_recall_fscore_support(y_true,y_pred,average='weighted')

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        myTable = PrettyTable(["Type","Accuracy","Precision","Recall","F1 - Score"])
        accuracy = round(np.sum(y_pred==y_true)/y_pred.shape[0],3)
        myTable.add_row(["Macro",accuracy,round(a_macro[0],3),round(a_macro[1],3),round(a_macro[2],3)])
        myTable.add_row(["Micro",accuracy,round(a_micro[0],3),round(a_micro[1],3),round(a_micro[2],3)])
        myTable.add_row(["Weighted",accuracy,round(a_weighted[0],3),round(a_weighted[1],3),round(a_weighted[2],3)])

        return myTable

model = KNNModel(10,"euclidean",2,train_data)
y_pred = []
y_true = []

outputs = model.inference(val_data)
y_pred = outputs
y_true = model.mapper_str_to_int_vectorize(val_data[:,3])

print(model.validation_metrics_table(y_true,y_pred))

END
)

# Run the Python script
python3 -c "$python_script"
