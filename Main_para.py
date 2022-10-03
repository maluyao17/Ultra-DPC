"""
Version of robustness detection.(Dpcpp_barch)
"""
from sklearn import metrics
from Dataprocessing import load_Data, load_Data2, get_data, get_data2, get_emnist
# from Dpcpp import Dpcpp
# from Dpcpp_ import Dpcpp
from Dpcpp_batch import Dpcpp
import numpy as np
import matplotlib.pyplot as plt


def plot_paraS(x, ari, ami, nmi, name):
    plt.figure(figsize=[6.40, 5.60])
    plt.plot(x, ari, color='r', marker='o', linestyle='-', alpha=1, label='ARI')
    plt.plot(x, ami, color='g', marker='o', linestyle='-', alpha=0.75, label='AMI')
    plt.plot(x, nmi, color='b', marker='o', linestyle='-', alpha=0.5, label='NMI')
    # plt.ylim([0, 0.7])
    plt.xlabel('para s')
    plt.ylabel('ARI/AMI/NMI')
    plt.legend()
    plt.title('ARI/AMI/NMI on '+name)

def plot_paraP(x, ari, ami, nmi, name):
    plt.figure(figsize=[6.40, 5.60])
    plt.plot(x, ari, color='r', marker='o', linestyle='-', alpha=0.5, label='ARI')
    plt.plot(x, ami, color='g', marker='o', linestyle='-', alpha=0.5, label='AMI')
    plt.plot(x, nmi, color='b', marker='o', linestyle='-', alpha=0.5, label='NMI')
    # plt.ylim([0, 0.8])
    plt.xlabel('para p')
    plt.ylabel('ARI/AMI/NMI')
    plt.legend()
    plt.title('ARI/AMI/NMI on '+name)

def plot_paraR(x, ari, ami, nmi, name):
    plt.figure(figsize=[6.40, 5.60])
    plt.plot(x, ari, color='r', marker='o', linestyle='-', alpha=0.5, label='ARI')
    plt.plot(x, ami, color='g', marker='o', linestyle='-', alpha=0.5, label='AMI')
    plt.plot(x, nmi, color='b', marker='o', linestyle='-', alpha=0.5, label='NMI')
    plt.ylim([0, 0.8])
    plt.xlabel('para r')
    plt.ylabel('ARI/AMI/NMI')
    plt.legend()
    plt.title('ARI/AMI/NMI on '+name)

file_Name = './datasets/emnist-digits.mat'
s = [ 6000]
p = [100, 200, 300, 400, 500, 600]
l = [100]
r = [150]
for h in range(len(s)):
    for x in range(len(p)):
        for t in range(len(l)):
            for f in range(len(r)):
                print(file_Name)
                # data, label = load_Data(file_Name)
                # data, label = get_data(file_Name)
                data, label = get_emnist(file_Name)
                # for i in range(10):
                opreator = Dpcpp(data, label, s[h], p[x], l[t], r[f])
                sample_label, time = opreator.fit()
                sample_label = np.array(sample_label, dtype=np.float64)
                org_label = label.astype(np.float64)
                ARI = metrics.adjusted_rand_score(org_label, sample_label)
                ARI_arr.append(ARI)
                AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
                AMI_arr.append(AMI)
                NMI = metrics.normalized_mutual_info_score(org_label, sample_label)
                NMI_arr.append(NMI)
                # print('s=', s[h], ' p=', p[x], ' l=', l[r])
                print('total time=', time)
                print("Adj. Rand Index Score=", ARI)
                print("Adj. Mutual Info Score=", AMI)
                print("Norm Mutual Info Score=", NMI)
plot_paraP(p, ARI_arr, AMI_arr, NMI_arr,'emnist-digits')
# plot_paraS(s, ARI_arr, AMI_arr, NMI_arr,'emnist-digits')
# plot_paraR(r, ARI_arr, AMI_arr, NMI_arr,'mnist')
plt.show()
