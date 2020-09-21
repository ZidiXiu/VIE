# VIE: Variational Inference for Extremals

Code for Variational Disentanglement for Rare Event Modeling (https://arxiv.org/abs/2009.08541)

## Model
We propose a
variational disentanglement approach to semi-parametrically
learn from rare events in heavily imbalanced classification
problems.

![Estimation of the tail](figures/tail_illustration.pdf)
Left: Distribution of a two-dimensional latent
space z where the long tail associates with higher risk. Right:
Tail estimations with different schemes for the long-tailed
data in one-dimensional space. EVT provides more accurate
characterization comparing to other mechanisms.


### Prerequisites

The algorithm is built with:

* Python (version 3.7 or higher)
* Numpy (version 1.16 or higher)
* PyTorch (version 1.13.1)

Clone the repository, e.g.:
```
git clone https://github.com/ZidiXiu/VIE.git
```


### Running the Binary Simulation Dataset

Here we present a toy synthetic dataset which enjoys a long-tailed behaviour in the latent space. 

```
python train train_VIE_simulationDL.py --batch-size 200 --epochs 500 
```

### Running the SLEEP Dataset

[SLEEP](https://sleepdata.org/datasets/shhs) A subset of the Sleep Heart Health Study (SHHS), a multi-center cohort study implemented by the National Heart Lung & Blood Institute to determine the cardiovascular and other consequences of sleep-disordered breathing. The dataset includes 5026 patients and 206 covariates.

```
python train train_VIE_SLEEP.py
```

## Acknowledgments

When building the VIE framework, we refenreced the following sources: 
* [Fenchel Min-max](https://github.com/Hanjun-Dai/cvb)
* [IAF](https://www.ritchievink.com/blog/2019/11/12/another-normalizing-flow-inverse-autoregressive-flows/)
* [Data Preprocessing](https://github.com/paidamoyo/survival_cluster_analysis)

