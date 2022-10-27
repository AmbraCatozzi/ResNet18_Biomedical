# Biomedical image classification via dynamically early stopped artificial neural network
The following code is based on the original scripts present in https://github.com/MedMNIST/MedMNIST, in particular this new code is designed to work only with PyTorch. The main differences from the original one are: the introduction of the Early Stopping technique and the implementation of other methods based on the learning rate annealing and the incremental Mini-Batch size. The complete dataset descripitions and references are reported in the previous link.

# Code Structure
* [`medmnist/`](medmnist/):
    * [`dataset.py`](medmnist/dataset.py): PyTorch datasets and dataloaders of MedMNIST.
    * [`evaluator.py`](medmnist/evaluator.py): Standardized evaluation functions.
    * [`info.py`](medmnist/info.py): Dataset information `dict` for each subset of MedMNIST.
* [`CT_OCT_experiments`](CT_OCT_experiments/):
    * [`CT_resnet18.ipynb`](CT_OCT_experiments/CT_resnet18.ipynb): Main script with training and evaluation of ResNet18 model on the MedMNIST2D OrganSMNIST for multi-class classification on abdominal CT images.
    * [`OCT_resnet18.ipynb`](CT_OCT_experiments/OCT_resnet18.ipynb):  Main script with training and evaluation of ResNet18 model on the MedMNIST2D OCTMNIST for multi-class classification on retinal OCT images.
    * [`models.py`](CT_OCT_experiments/models.py): Code with models.
* [`setup.py`](setup.py): To install `medmnist` as a module.

# Installation, Requirements and Dataset
The following instruction are from the original repository https://github.com/MedMNIST/MedMNIST where you can find also the datasets and the command line tools.
In our code we have chosen to modify the subdivision phase of the dataset to ensure a random and varied selection of samples.

Setup the required environments and install `medmnist` as a standard Python package from [PyPI](https://pypi.org/project/medmnist/):

    pip install medmnist

Or install from source:

    pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Check whether you have installed the latest [version](medmnist/info.py):

    >>> import medmnist
    >>> print(medmnist.__version__)

The code requires only common Python environments for machine learning. Basically, it was tested with
* Python 3 (>=3.6)
* PyTorch\==1.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2, Pillow\==8.0.1, fire, scikit-image

Higher (or lower) versions should also work (perhaps with minor modifications). 

# Warning
These datasets are NOT intended for clinical use.

# License and Citation

The orginal code is under [Apache-2.0 License](./LICENSE).

The MedMNIST dataset is licensed under *Creative Commons Attribution 4.0 International* ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
If you find this project useful in your research, please cite the following papers:

    Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification". arXiv preprint arXiv:2110.14795, 2021.

    Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

or using the bibtex:

    @article{medmnistv2,
        title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
        author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
        journal={arXiv preprint arXiv:2110.14795},
        year={2021}
    }
     
    @inproceedings{medmnistv1,
        title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
        author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
        booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
        pages={191--195},
        year={2021}
    }
    @article{franchini2022biomedical,
        title={Biomedical Image Classification via Dynamically Early Stopped Artificial Neural Network},
        author={Franchini, Giorgia and Verucchi, Micaela and Catozzi, Ambra and Porta, Federica and Prato, Marco},
        journal={Algorithms},
        volume={15},
        number={10},
        pages={386},
        year={2022},
        publisher={MDPI}
}

Please also cite source data paper(s) of the MedMNIST subset(s) as per the [description](https://medmnist.github.io/).
