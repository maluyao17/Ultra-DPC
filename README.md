## DPC++ : Density Peak Clustering based on Sampling

### Runtime environment

* Python:3.9.7
* scikit-learn:0.24.2
* scipy:1.7.1
* numpy:1.20.3
* matplotlib:3.4.3
* seaborn:0.11.2
* h5py:3.2.1

### Composition of folders

* Main.py: Official version without parameter debugging.
* Main_para.py: Version of robustness detection.
* Main_debug.py: Parameter debugging version.
* Dpcpp.py: DPC++ Laplacian matrix construction based on the pxN matrix version.
* DPCpp_.py: DPC++ Laplacian matrix construction based on the Nxp matrix version.
* Dpcpp_batch.py: DPC++ for Nxp large-scale Laplacian matrix calculation in batches.
* Dataprocessing.py:  Functions related to data normalization and unitary processing operations.

### Result:

![modes](.\figure\modes.png)![dpc++ s2](.\figure\dpc++ s2.png)