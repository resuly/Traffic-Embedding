### The embedding model for categorical data analysis in transport.

This repository contains the related codes of the paper 'Revealing the hidden features in traffic prediction via entity embedding'. Please see this blog for more information:http://resuly.me/2020/02/18/embedding-in-transport/



The main code located in `model` folder, and the visualization works can be found in `visualization`.

To run the embedding model, you will need to install PyTorch environment and run the following command:

`python train.py --model EM`

See the results in `experiments/EM`


If you think this is helpful to your research, please consider citing our work:

`
@article{wang2019revealing,
  title={Revealing the hidden features in traffic prediction via entity embedding},
  author={Wang, Bo and Shaaban, Khaled and Kim, Inhi},
  journal={Personal and Ubiquitous Computing},
  pages={1--11},
  year={2019},
  publisher={Springer}
}
`

