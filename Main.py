"""
Official version without parameter debugging.(Dpcpp_.py)
"""
from sklearn import metrics
from Dataprocessing import load_Data, load_Data2, get_data, get_data2
# from Dpcpp import Dpcpp
from Dpcpp_ import Dpcpp
# from Dpcpp_batch import Dpcpp
import numpy as np

file_Name = './datasets/s2.csv'
s = 3000
p = 300
l = 60
print(file_Name)
data, label = load_Data(file_Name)
opreator = Dpcpp(data, label, s, p, l)
sample_label, time = opreator.fit()
sample_label = np.array(sample_label, dtype=np.float64)
org_label = label.astype(np.float64)
ARI = metrics.adjusted_rand_score(org_label, sample_label)
AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
NMI = metrics.normalized_mutual_info_score(org_label, sample_label)
print('s=', s, ' p=', p, ' l=', l)
print('total time=', time)
print("Adj. Rand Index Score=", ARI)
print("Adj. Mutual Info Score=", AMI)
print("Norm Mutual Info Score=", NMI)
str__ = "\nfile_name:" + str(file_Name) + "\ns:" + str(s) + ",p:" + str(p) + ",l:" + str(l)  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
with open("dpc_data.txt", "a+") as f:
    f.write(str__)
