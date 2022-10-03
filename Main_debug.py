"""
Parameter debugging version.(Dpcpp_batch.py)
"""
from sklearn import metrics
from Dataprocessing import load_Data, load_Data2, get_data, get_data2, get_emnist
# from Dpcpp import Dpcpp
# from Dpcpp_ import Dpcpp
from Dpcpp_batch import Dpcpp
import numpy as np

file_Name = './datasets/R15.csv'
# file_Name = './datasets/covtype.csv'
s = [400, 500]
p = [200, 300, 350]
l = [5, 10]
for h in range(len(s)):
    for x in range(len(p)):
        for e in range(len(l)):
            print(file_Name)
            data, label = load_Data(file_Name)
            # data, label = get_data(file_Name)
            # data, label = get_emnist(file_Name)
            # for i in range(10):
            opreator = Dpcpp(data, label, s[h], p[x], l[e])
            sample_label, time = opreator.fit()
            sample_label = np.array(sample_label, dtype=np.float64)
            org_label = label.astype(np.float64)
            ARI = metrics.adjusted_rand_score(org_label, sample_label)
            AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
            NMI = metrics.normalized_mutual_info_score(org_label, sample_label)
            print('s=', s[h], ' p=', p[x], ' l=', l[e])
            print('total time=', time)
            print("Adj. Rand Index Score=", ARI)
            print("Adj. Mutual Info Score=", AMI)
            print("Norm Mutual Info Score=", NMI)
            str__ = "\nfile_name:" + str(file_Name) + "\ns:" + str(s[h]) + ",p:" + str(p[x]) + ",l:" + str(l[e])  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
            with open("dpc_R15.txt", "a+") as f:
                f.write(str__)
