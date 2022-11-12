"""
Official version without parameter debugging.(Dpcpp_.py)
"""
from sklearn import metrics
from Dataprocessing import get_emnist, load_Data, load_Data2, get_data, get_data2
# from Dpcpp import Dpcpp
from Dpcpp_ import Dpcpp
# from Dpcpp_batch import Dpcpp
import numpy as np

# ARI_avg = 0.0
# AMI_avg = 0.0
# NMI_avg = 0.0
file_Name = './Datasets/mnist_2D.csv'
s = 8000
p = 200
l = 50
print(file_Name)
data, label = load_Data(file_Name)
# data, label = get_data(file_Name)
# data, label = get_emnist(file_Name)

# for i in range(10):
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
# ARI_avg += ARI
# AMI_avg += AMI
# NMI_avg += NMI
str__ = "\nfile_name:" + str(file_Name) + "\ns:" + str(s) + ",p:" + str(p) + ",l:" + str(l)  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
with open("dpc_new.txt", "a+") as f:
    f.write(str__)

# ARI_avg = ARI_avg/10.0
# AMI_avg = AMI_avg/10.0
# NMI_avg = NMI_avg/10.0
# print("Average Adj. Rand Index Score=", ARI_avg)
# print("Average Adj. Mutual Info Score=", AMI_avg)
# print("Average Norm Mutual Info Score=", NMI_avg)

