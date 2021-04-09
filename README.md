
<p align="center">
    <img src="jupyterbook/ml4floods_banner.png" alt="awesome ml4floods" width="300">
</p>

_ML4Floods: an ecosystem of data, models and code pipelines to tackle flooding with ML_

This repository contains an end-to-end ML pipeline for flood extent estimation: from data preprocessing, model training, model deployment to visualization.

Install the package:

```bash
pip install git+https://github.com/spaceml-org/ml4floods#egg=ml4floods
```

These tutorials may help you explore the datasets and models:
* [Project rationale](http://trillium.tech/ml4floods/content/intro/introduction.html).
* [Data Preprocessing](https://github.com/spaceml-org/ml4floods/tree/main/notebooks/data/preprocessing)
* [ML-based flood segmentation models](http://trillium.tech/ml4floods/content/ml_overview.html)
    * [Training](http://trillium.tech/ml4floods/content/ml4ops/HOWTO_Train_models.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
    * Inference on [new data](http://trillium.tech/ml4floods/content/ml4ops/HOWTO_Run_Inference_on_new_data.html) (a Sentinel-2 image) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)
    * [Perf metrics](http://trillium.tech/ml4floods/content/ml4ops/HOWTO_performance_metrics_workflow.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)


If you find this work useful please cite:

```
@article{mateo-garcia_towards_2021,
	title = {Towards global flood mapping onboard low cost satellites with machine learning},
	volume = {11},
	issn = {2045-2322},
	doi = {10.1038/s41598-021-86650-z},
	number = {1},
	urldate = {2021-04-01},
	journal = {Scientific Reports},
	author = {Mateo-Garcia, Gonzalo and Veitch-Michaelis, Joshua and Smith, Lewis and Oprea, Silviu Vlad and Schumann, Guy and Gal, Yarin and Baydin, Atılım Güneş and Backes, Dietmar},
	month = mar,
	year = {2021},
	pages = {7249},
}
```
