# Pytorch 1.0.0 code for: Weakly Supervised Object Localization using Min-Max Entropy: an Interpretable Framework

# Reproducibility:

We took a particular care to the reproducibility of the code.
* The code is reproducible under [Pytorch reproducibility terms](https://pytorch.org/docs/stable/notes/randomness.html).
> Completely reproducible results are not guaranteed across PyTorch releases, individual commits or different platforms.
 Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.
* The code is guaranteed to be reproducible over the same device INDEPENDENTLY OF THE NUMBER OF WORKERS (>= 0). You 
have to use a seed in order to obtain the same results over two identical runs. See [./reproducibility.py](./reproducibility.py)
* Samples can be preloaded in memory to avoid disc access. See [./loader.PhotoDataset()](./loader.py). DEFAULT SEED 
IS 0 which we used in all our experiments.
* Samples can be preloaded AND preprocessed AND saved in memory (for inference, i.e., test). However, for large 
dataset, and when the pre-processing aims at increasing the size of the data (for instance, upsampling), this is to 
be avoided. See [./loader.PhotoDataset()](./loader.py)
* We decorated *sensitive* operations that use random generators by a fixed seed. This allowed more flexibility in 
terms of reproducibility. For instance, if you decide to switch off pre-processing samples of a dataset (given that 
such operation relies heavily on random generators) and do the processing on the fly, such a decorator seed allows the 
state of the random generators in the following code to be independent of the results of this switch. This is a 
work-around. In the future, we consider more clean, and easy way to make the operations that depend on random 
generator independent of each other.

All the experiments, splits generation were achieved using seed 0.
Please report any issue with reproducibility.

# Download datasets:

* GlaS: You can use [./download-glas-dataset.sh](./download-glas-dataset.sh).
* Caltech-UCSD Birds-200-2011: You can use [./download-caltech-ucsd-birds-200-2011-dataset.sh](./download-caltech-ucsd-birds-200-2011-dataset.sh)

You find the splits in [./folds](./folds). The code that generated the splits is [./create_folds.py](./create_folds.py).

# Requirements:
We use [Pytorch 1.0.0](https://pytorch.org/) and [Python 3.7.0](https://www.python.org). For installation, see [./dependencies](
./dependencies) for a way on how to install the requirements within a virtual environment. 
 
# Paths:
We hard-coded some paths (to the data location). For anonymization reasons, we replaced them with fictive paths. 
So, they won't work for you. A warning will be raised with an indication to the issue. Then, the code exits. Something 
like this:
```python
warnings.warn("You are accessing an anonymized part of the code. We are going to exit. Come here and fix this "
                  "according to your setup. Issue: absolute path to Caltech-UCSD-Birds-200-2011 dataset.")
```

# Configuration used in the paper:
The yaml files in [./config_yaml](./config_yaml) are used for each dataset.

# To tun the code:
```bash
python train_deepmil.py --cudaid your_cuda_id --yaml basename_your_yaml_file
```
You can override the values of the yaml file using command line:
```bash
python train_deepmil.py --cudaid 1 --yaml glas-no-erase.yaml --kmin 0.09 --kmax 0.09 --dout 0.0 --epoch 2 \
--nbrerase 0 --epocherase 1 --stepsize 40 --bsize 8  --modalities 5 --lr 0.001 --fold 0 --wdecay 1e-05 --alpha 0.2
```
See all the keys that you can override using the command line in  [tools.get_yaml_args()](./tools.py).
