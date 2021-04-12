### The embedding model for categorical data analysis in transport.

### Introduction

This is an open-source (MIT) [Pytorch](https://github.com/pytorch/pytorch) based code repository (feature embedding) for the following paper:

 "Wang, B., Shaaban, K. and Kim, I., 2019. Revealing the hidden features in traffic prediction via entity embedding. *Personal and Ubiquitous Computing*, pp.1-11."

The feature embedding is designed to represent discreate (or categorical) variables in traffic forecasting tasks. More information can be found at http://resuly.me/2020/02/18/embedding-in-transport/

![](http://resuly.me/img/in_post/2020/embedding/id_all_1-1_-_no_numbers.svg)

### Usage

The main code located in the `model` folder and the visualization works can be found in `visualization`.

To run the embedding model, you will need to install PyTorch environment and run the following command:

`python train.py --model EM`

See the results in `experiments/EM`

### Citation


If you think this is helpful to your research, please consider citing our work:

```
@article{wang2019revealing,
  title={Revealing the hidden features in traffic prediction via entity embedding},
  author={Wang, Bo and Shaaban, Khaled and Kim, Inhi},
  journal={Personal and Ubiquitous Computing},
  pages={1--11},
  year={2019},
  publisher={Springer}
}
```
