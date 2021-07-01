# Improved CNN for Classifying Signals in Fault Diagnosis


[PAGE IN PROGRESS]

This repository is supplementary to our paper "An Improved Wide-Kernel CNN for Classifying Multivariate Signals in Fault Diagnosis" for reproducing our proposed 
models and their respective performance. 

### Abstract
Deep Learning (DL) provides considerable opportunities for increased efficiency and performance in fault diagnosis. The ability of DL methods for automatic 
feature extraction can reduce the need for time-intensive feature construction and prior knowledge on complex signal processing. In this paper, we propose two 
models that are built on the Wide-Kernel Deep Convolutional Neural Network (WDCNN) framework to improve performance of classifying fault conditions using 
multivariate time series data, also with respect to limited and/or noisy training data. In our experiments, we use the renowned benchmark dataset from the Case 
Western Reserve University (CWRU) bearing experiment [1] to assess our models’ performance, and to investigate their usability towards large-scale applications 
by simulating noisy industrial environments. Here, the proposed models show an exceptionally good performance without any preprocessing or data augmentation and 
outperform traditional Machine Learning applications as well as state-of-the-art DL models considerably, even in such complex multi-class classifica- tion tasks. 
We show that both models are also able to adapt well to noisy input data, which makes them suitable for condition- based maintenance contexts. Furthermore, we 
investigate and demonstrate explainability and transparency of the models which is particularly important in large-scale industrial applications.

### Citation
When using our code, please cite our paper as follows:
```
@article{hoogen2020improvedWDCNN,
  title={An Improved Wide-Kernel CNN for Classifying Multivariate Signals in Fault Diagnosis},
  author={van den Hoogen, Jurgen and Bloemheuvel, Stefan and Atzmueller, Martin},
  journal={ICDMW},
  year={2020}
}
```

### Requirements
Usage of our code requires many packages to be installed on your machine. The most important packages are listed below:
* Numpy
* Tensorflow
* Keras
* Multivariate-cwru

### Data
The data is collected from the [Case Western Reserve University][cwru] Bearing webpage to classify bearing fault conditions based on multivariate signals.
Please consult the [`Multivariate CWRU`][multivariate_cwru] package description to extract and preprocess the data. 

### Usage
The experiment files can be found in the `Bearing fault experiment` folder. Both the `"import_file.py"` and `"Utils.py"` need to be executed before deploying one of the provided models. 





[cwru]: <https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website>
[multivariate_cwru]: <https://github.com/JvdHoogen/multivariate_cwru>
