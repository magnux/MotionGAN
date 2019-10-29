# MotionGAN

This is repository of the code used in the paper ["Human Motion Prediction via Spatio-Temporal Inpainting"](https://arxiv.org/abs/1812.05478).
The repository is a framework to build GAN models for motion prediction.
Many parts of the code were taken from the repo of [Martinez et al](https://github.com/una-dinosauria/human-motion-prediction)

## Requirements

First you have to set up an environment with the proper requirements.
For that we recommend using virtualenv, but conda or plain python would work too.
In any case, install a python 2.7 environment, then install the requirements with:

`pip install -r requirements.txt`

## Datasets

Many datasets are implemented but only two are tested and known to be working with the last version of the model: 
- Human3.6 : The positions D3 version (not universal, not mono). Processing script: utils/human36_skels_to_h5.py
- MSRC12: The standard oribinal version. Processing script: utils/msrc12_skels_to_h5.py

Download the datasets, then process them with the appropriate script in the utils folder.
Finally, put them in the data folder (default), or define a path to the location of the processed data in the config.

## Training the model

The fast route to obtain the model is to run:

python train.py -config_file motiongan_v7n_alldisc_h36

To restart the training if needed:

python train.py -save_path save/motiongan_v7n_alldisc_h36

### Understanding the implementation
The file models/motiongan.py is the main file of the repository.
In it we can find the MotionGAN class, a model factory that builds a customized model according to a specific config.
Then we have many children classes MotionGANv*, each implements a different network architecture for the inference.
- v1: a simple resnet model.
- v3: a simple feedforward model.
- v5: an autoregressive model.
- v7: the proposed network in the first version of the paper.
- v7n: the latest version with injected noise.

There are many more models, hybrid rnn/cnn models, etc, we encourage the reader to take a look in the file.

The config used for the best results in the paper is motiongan_v7n_alldisc_h36.
The baseline nogan model is motiongan_v7n_nogan_h36.
Check the [config file](#the-config-file) section for more details on the config file.

## Testing the model

The file test.py is a multipurpose script that can perform many different tests over the trained models.

To print sample results:

python test.py -model_path save/motiongan_v7n_alldisc_h36 -test_mode show_images

To perform the PS Metric on a model:

python test.py -model_path save/motiongan_v7n_alldisc_h36 -test_mode paper_metrics

Read over the script for the full range of options.

## The config file

The file configs/base_config.py has all the parameters and default values. 
The other configs override such values to create custom models and training setups.
It is recommended to read carefully the base options before creating a custom model.
Be aware that many options may be mutually exclusive.

## Misc goodies
There are many miscelaneous code that may be useful or interesting for further work, among them we highlight:

Layers:
- [edm.py](layers/edm.py): Euclidean distance matrix computation in Keras.
- [noise.py](layers/noise.py): Noise injection implementation in Keras.
- [seq_transform.py](layers/seq_transform.py): A series of layers to transform the sequences on the fly, modifiying the input to the network and then reverting the change on the output.
Utils:
- [npangles.py](utils/npangles.py): a tensorized version of the angles library in [Martinez et al](https://github.com/una-dinosauria/human-motion-prediction)
- [tfangles.py](utils/tfangles.py): a tensorflow version of the angles library.
- [seq_utils.py](utils/seq_utils.py): a library with many useful functions to process the sequences in numpy, such as transforming the skeleton coordinates to angles representation etc.
- [scoping.py](utils/scoping.py): a scoping library useful to obtain neat graphs in keras.


## Citation

@article{ruiz2018human,
  title={Human Motion Prediction via Spatio-Temporal Inpainting},
  author={Ruiz, Alejandro Hernandez and Gall, Juergen and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:1812.05478},
  year={2018}
}

## License
MIT License:

Copyright 2019 Alejandro Hernandez Ruiz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.