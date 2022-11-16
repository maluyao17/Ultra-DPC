from sklearn import metrics
from Dataprocessing import get_8m, load_Data, load_Data2, get_data, get_data2
import Dpcpp_for8m
from Dpcpp_for8m import Dpcpp
# from Dpcpp_big import Dpcpp
import numpy as np
import scipy.io as sio
import datetime
import gc


# ARI_avg = 0.0
# AMI_avg = 0.0
# NMI_avg = 0.0
file_Name = ['E:/mnist8m/mnist1q.mat','E:/mnist8m/mnist2q.mat',
                'E:/mnist8m/mnist3q.mat','E:/mnist8m/mnist4q.mat',
                'E:/mnist8m/mnist5q.mat','E:/mnist8m/mnist6q.mat',
                'E:/mnist8m/mnist7q.mat','E:/mnist8m/mnist8q.mat',
                'E:/mnist8m/mnist9q.mat','E:/mnist8m/mnist10q.mat',
                'E:/mnist8m/mnist11q.mat','E:/mnist8m/mnist12q.mat',
                'E:/mnist8m/mnist13q.mat','E:/mnist8m/mnist14q.mat',
                'E:/mnist8m/mnist15q.mat','E:/mnist8m/mnist16q.mat']

s = [20000]
p = [400]
l = [5]
for h in range(len(s)):
    for x in range(len(p)):
        for e in range(len(l)):
            starttime = datetime.datetime.now()
            label = []
            data = []
            index_return = []
            data_to_kms = []
            starttime = datetime.datetime.now()
            for i in range(0, 16):
                print(file_Name[i])
                data_tmp, label_tmp = get_8m(file_Name[i])
                # label_tmp = label_tmp.tolist()
                if any(label)==False:
                    label = label_tmp
                    tmp_index, data_to_kms_all, kms_center = Dpcpp_for8m.get_prototype(data_tmp, s[h], p[x])
                    # data = data_tmp

                else:
                    label = np.r_[label, label_tmp]
                    tmp_index = Dpcpp_for8m.get_alldist(data_tmp, p[x] ,tmp_index, kms_center)

            opreator = Dpcpp(label, data_to_kms_all, tmp_index, s[h], p[x], l[e], 500000)
            sample_label = opreator.fit()
            endtime = datetime.datetime.now()
            time = endtime - starttime
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
            # ARI_avg += ARI
            # AMI_avg += AMI
            # NMI_avg += NMI
            str__ = "\nfile_name:" + str(file_Name) + "\ns:" + str(s[h]) + ",p:" + str(p[x]) + ",l:" + str(l[e])  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
            with open("dpc_8m.txt", "a+") as f:
                f.write(str__)


