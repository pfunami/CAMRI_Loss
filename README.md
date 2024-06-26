# CAMRI Loss: Improving Recall of a Specific Class without Sacrificing Accuracy

This repository is the implementation of Class-sensitive additive Angular MaRgIn Loss (CAMRI Loss)
by TensorFlow.

In multi-class classification problems, CAMRI loss can improve recall of the specific class(es) without sacrificing
accuracy comparing to cross-entropy loss.

Our paper & poster corresponding to this implementation are followings:
- [Journal](https://www.jstage.jst.go.jp/article/transinf/E106.D/4/E106.D_2022EDP7200/_article/-char/en) (IEICE Transactions on Information and Systems, Volume E106.D (2023) Issue 4 Pages 523-537, 2023)
- [Conference Paper](https://ieeexplore.ieee.org/document/9892108) (IJCNN ’22, July 18-23, 2022, Padua, Italy.)
- [Preprint](https://arxiv.org/abs/2209.10920) (The author version of the paper accepted by IJCNN ’22, July 18-23,
2022, Padua, Italy.)
- [Poster](https://github.com/pfunami/CAMRI_Loss/blob/master/doc/ijcnn_camri.pdf)

## Abstract

From our paper
> In real-world applications of multi-class classification models, misclassification in an important class (e.g., stop sign) can be significantly more harmful than other classes (e.g., speed limit). In this paper, we propose a loss function that can improve the recall of an important class while maintaining the same level of accuracy as the case using cross-entropy loss. For our purpose, we need to make the separation of the important class better than the other classes. However, existing methods that give a class-sensitive penalty for cross-entropy loss do not improve the separation. On the other hand, the method that gives a margin to the angle between the feature vectors and the weight vectors of the last fully connected layer corresponding to each feature can improve the separation. Therefore, we propose a loss function that can improve the separation of the important class by setting the margin only for the important class, called Class-sensitive Additive Angular Margin Loss (CAMRI Loss). CAMRI loss is expected to reduce the variance of angles between features and weights of the important class relative to other classes due to the margin around the important class in the feature space by adding a penalty to the angle. In addition, concentrating the penalty only on the important classes hardly sacrifices the separation of the other classes. Experiments on datasets: CIFAR-10, GTSRB, and AwA2 showed that the proposed method could improve up to 9% recall improvement on cross-entropy loss without sacrificing accuracy.

## How to use

### Example of possible environment

- Python 3.8
- TensorFlow 2.x
- conda 4.12.0

### Installation
Using conda env
1. Clone this repository `https://github.com/ProfFunami/CAMRI_Loss.git`
2. `$ cd CAMRI_Loss`
3. `$ conda create --name <ENV_NAME> --file conda_requirements.txt`
4. `$ conda activate <ENV_NAME>`

### Run training

1. `$ cd CAMRI_Loss/src`
2. `$ python train.py`

## Repository Structure

```angular2html
.
├── README.md
├── conda_requirements.txt
├── config.ini      # configuration file (e.g., important class, margin, scale, ...)
├── out             # examples of a trained model file and a log file
│   ├── log
│   │   └── cifar10_3cat_m4-64_scale16_seed0.csv
│   └── model
│       └── cifar10_3cat_m4-64_scale16_seed0.h5
└── src
    ├── archs.py    # define the model architecture
    ├── camri.py    # define CAMRI loss
    ├── eval.py     # callback function for each epoch
    └── train.py    # main code for training the model
```

## Citation

```angular2html
@INPROCEEDINGS{9892108,
    author={Nishiyama, Daiki and Fukuchi, Kazuto and Akimoto, Youhei and Sakuma, Jun},
    booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},   
    title={CAMRI Loss: Improving Recall of a Specific Class without Sacrificing Accuracy}, 
    year={2022},  
    volume={},  
    number={}, 
    pages={1-8},
    doi={10.1109/IJCNN55064.2022.9892108}}
    
@article{Daiki NISHIYAMA20232022EDP7200,
  title={CAMRI Loss: Improving the Recall of a Specific Class without Sacrificing Accuracy},
  author={Daiki NISHIYAMA and Kazuto FUKUCHI and Youhei AKIMOTO and Jun SAKUMA},
  journal={IEICE Transactions on Information and Systems},
  volume={E106.D},
  number={4},
  pages={523-537},
  year={2023},
  doi={10.1587/transinf.2022EDP7200}
}
```

## Contact

Daiki Nishiyama (First Author)

- Affiliation: University of Tsukuba, Japan
- Mail: nsym [at] mdl.cs.tsukuba.ac.jp

## P.S.

Shortly after submitting the paper on CAMRI loss, my beloved car, CAMRY, was involved in a rear-end collision and was
scrapped 😭

In other words, when I proposed CAMRI loss, CAMRY was lost 😇
