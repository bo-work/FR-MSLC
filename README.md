# FR-MSLC：Malicious Traffic Detection from Noisy Label and Imbalanced Data



![FR-MSCL](https://github.com/bo-work/FR-MSLC/blob/main/overall2.png)



## Description: 

An official PyTorch implementation of the "FR-MSLC: Malicious Traffic Detection from Noisy Label and Imbalanced Data" paper.

## Main Results

Test in Improved-CICIDS-2017:

| Asym-20% | Asym-40% | Asym-60% | Asym-80% | Sym-20% | Sym-40% | Sym-60% | Sym-80% |
| :------: | :------: | :------: | :------: | :-----: | :-----: | :-----: | :-----: |
|  99.73   |  98.72   |  97.93   |   95.5   |  92.27  |  90.73  |  88.65  |  88.54  |

Test in Malicious-TLS-2023:

| Asym-20% | Asym-40% | Asym-60% | Asym-80% | Sym-20% | Sym-40% | Sym-60% | Sym-80% |
| :------: | :------: | :------: | :------: | :-----: | :-----: | :-----: | :-----: |
|  93.98   |  92.74   |  90.07   |  88.11   |  94.01  |  93.18  |  92.35  |  90.38  |

Test about unannotation attack:

| Avg Acc (%) | Acc of Patator  (%) | Avg Acc (%) | Acc of Banker (%) | Avg Acc (%) | Acc of Caphaw.A (%) |
| :---------: | :-----------------: | :---------: | :---------------: | :---------: | ------------------- |
|    99.14    |        99.28        |    93.94    |       19.77       |    90.57    | 100.00              |



## Requirements

pytorch = 2.0.0

cuda = 11.8

To install detailed requirements:

```setup
pip install -r requirements.txt
```

## Running

Before training the models please:

1. Put the datasets in the `/data` . Origian dataset is [ Improved-CICIDS-2017](https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html) and [Malicious-TLS-2023](https://github.com/gcx-Yuan/Malicious_TLS).
2. Pre_processing with `pyhton data_process.py`. Or download preprocessed data [here](seek for anonymous) and put them in the `/data`.
3. Feature representation leanrn with `python FRlenrning.py`. Or download preprocessed data [here](seek for anonymous) and put them in the `./results/models/deepre`.
4. Pretrained detection model with `python frmslc_mlp_pre.py` and just comment corrsponing dataset info for different dataset.  Or download pretrained model [here](seek for anonymous) and put them in the `./results/models/warmup`.
5. Train with `python frmslc_mlp.py`, and you can change every parameters in config_mlp.py


## Contributing

This repository is heavily based on [MSLC](https://github.com/WuYichen-97/Learning-to-Purify-Noisy-Labels-via-Meta-Soft-Label-Corrector) and [DSDIR](https://github.com/nku-ligl/DSDIR)
