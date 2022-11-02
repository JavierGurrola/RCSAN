# Arbitrary Scale Super-Resolution Neural Network Based on Residual Channel-Spatial Attention

This repository is for the RCSAN model proposed in the following paper:

[Javier Gurrola-Ramos](https://scholar.google.com.mx/citations?user=NuhdwkgAAAAJ&hl=es), [Oscar Dalmau](https://scholar.google.com.mx/citations?user=5oUOG4cAAAAJ&hl=es&oi=sra) and [Teresa E. Alarcón](https://scholar.google.com.mx/citations?user=gSUClZYAAAAJ&hl=es&authuser=1), ["Arbitrary Scale Super-Resolution Neural Network Based on Residual Channel-Spatial Attention"](https://ieeexplore.ieee.org/document/9906989), in IEEE Access, vol. 10, pp. 108697-108709, 2022, doi: [10.1109/ACCESS.2022.3211302](https://doi.org/10.1109/ACCESS.2022.3211302).

## Citation
If you use this paper work in your research or work, please cite our paper:

```
@ARTICLE{gurrola2022arbitrary,
  author={Gurrola-Ramos, Javier and Alarcón, Teresa E. and Dalmau, Oscar},
  journal={IEEE Access}, 
  title={Arbitrary Scale Super-Resolution Neural Network Based on Residual Channel-Spatial Attention}, 
  year={2022},
  volume={10},
  number={},
  pages={108697-108709},
  doi={10.1109/ACCESS.2022.3211302}
}

```
![RCSAN](https://github.com/JavierGurrola/RCSAN/blob/main/figs/model.png)

## Dependencies
- Python 3.6
- PyTorch 1.8.0
- Torchvision 0.9.0
- Numpy 1.19.2
- Pillow 8.1.2
- ptflops 0.6.4
- tqdm 4.50.2
- scikit-image 0.17.2
- PyYaml 5.3.1


## Training


Default parameters used in the paper are set in the ```config.yaml``` file:

```
base filters: 64
patch size: 48
batch size: 16
learning rate: 1.e-4
weight decay: 1.e-5
gamma decay: 0.5
step decay: 200
epochs: 1000
samples per epoch: 16000
scale range:
    - 1.1
    - 4.0
```

To train the model use the following command:

```python main_train.py```

## Test and pre-trained models

Pre-trained models are available in this [link](https://drive.google.com/drive/folders/1FXHVRwYbwez7b3f0f_feE692aBWb2myr?usp=sharing). You can download and put them in the pre-trained folder.

To test the model use the following command:

```python main_test.py```


## Contact

If you have any question about the code or paper, please contact francisco.gurrola@cimat.mx .
