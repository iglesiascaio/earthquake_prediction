---
model_id: Toto-Open-Base-1.0
tags:
- time-series-forecasting
- foundation models
- pretrained models
- time series foundation models
- time series
- time-series
- transformers
- forecasting
- safetensors
- observability
paper:
- - Link to Paper
datasets:
- Salesforce/GiftEvalPretrain
- autogluon/chronos_datasets
leaderboards:
- GiftEval (if results are public)#TODO(Anna) check how to do that
- BOOM (if results are public)#TODO(Anna) check how to do that
license: apache-2.0
pipeline_tag: time-series-forecasting
---
# Toto-Open-Base-1.0

Toto (Time Series Optimized Transformer for [Observability](https://www.datadoghq.com/knowledge-center/observability/)) is a **state-of-the-art** time-series foundation model designed for multi-variate time series forecasting, emphasizing observability metrics. Toto efficiently handles high-dimensional, sparse, and non-stationary data commonly encountered in observability scenarios.

<div style="width: 100%; margin: auto; padding: 1rem;">
  <img src="figures/rankings.png" alt="model ranking" style="width: 100%; height: auto;" />
  <em style="display: block; margin-top: 0.5rem; text-align: center;">
    The average rank of Toto compared to the runner-up models on both the <a href="https://huggingface.co/spaces/Salesforce/GIFT-Eval">GIFT-Eval</a> and <a href="https://huggingface.co/datasets/Datadog/BOOM">BOOM</a> benchmarks (as of May 19, 2025).
  </em>
</div>

---

## ✨ Key Features

- **Zero-Shot Forecasting**: Perform forecasting without fine-tuning on your specific time series.
- **High-Dimension Multi-Variate Support**: Efficiently process multiple variables using Proportional Factorized Space-Time Attention.
- **Decoder-Only Transformer Architecture**: Support for variable prediction horizons and context lengths.
- **Probabilistic Predictions**: Generate both point forecasts and uncertainty estimates using a Student-T mixture model.
- **Extensive Pretraining on Large-Scale Data**: Trained on over 2 trillion time series data points, the largest pretraining dataset for any open-weights time series foundation model to date.
- **Tailored for Observability Metrics with State-of-the-Art Performance** on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [BOOM](https://huggingface.co/datasets/Datadog/BOOM).


<div style="width: 100%; margin: auto; padding: 1rem;">
  <img src="figures/architecture.png" alt="model architecture" style="width: 100%; height: auto;" />
  <em style="display: block; margin-top: 0.5rem; text-align: center;">
    Overview of Toto-Open-Base-1.0 architecture.
  </em>
</div>

---

## 📚 Training Data Summary

- **Observability Metrics:** ~1 trillion points from Datadog internal systems (no customer data)
- **Public Datasets:**
  - [GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
  - [Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)
- **Synthetic Data:** ~1/3 of training data



---

## ⚡ Quick Start: Model Inference

Inference code is available on [GitHub](https://github.com/DataDog/toto).

### Installation

```bash
# Clone the repository
git clone https://github.com/DataDog/toto.git
cd toto

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Inference Example

Here's how to quickly generate forecasts using Toto:

⚠️ In our study, we take the **median** across 256 samples to produce a point forecast. This tutorial previously used the **mean** but has now been updated.

```python
import torch
from data.util.dataset import MaskedTimeseries
from inference.forecaster import TotoForecaster
from model.toto import Toto

DEVICE = 'cuda'

# Load pre-trained Toto model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(DEVICE)

# Optional: compile model for enhanced speed
toto.compile()

forecaster = TotoForecaster(toto.model)

# Example input series (7 variables, 4096 timesteps)
input_series = torch.randn(7, 4096).to(DEVICE)
timestamp_seconds = torch.zeros(7, 4096).to(DEVICE)
time_interval_seconds = torch.full((7,), 60*15).to(DEVICE)

inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    id_mask=torch.zeros_like(input_series),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

# Generate forecasts for next 336 timesteps
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,
    num_samples=256,
    samples_per_batch=256,
)

# Access results
median_prediction = forecast.median
prediction_samples = forecast.samples
lower_quantile = forecast.quantile(0.1)
upper_quantile = forecast.quantile(0.9)
```

For detailed inference instructions, refer to the [inference tutorial notebook](https://github.com/DataDog/toto/blob/main/toto/notebooks/inference_tutorial.ipynb).

### Performance Recommendations
- ### **For optimal speed and reduced memory usage, install [xFormers](https://github.com/facebookresearch/xformers) and [flash-attention](https://github.com/Dao-AILab/flash-attention). Then, set `use_memory_efficient` to `True`.**

---

### 💾 Available Checkpoints

| Checkpoint | Parameters | Config | Size | Notes |
|------------|------------|--------|------|-------|
| [Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/main/model.safetensors) | 151M | [Config](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/main/config.json) | 605 MB | Initial release with SOTA performance |




## 🔗 Additional Resources

- **[Research Paper](https://arxiv.org/abs/2505.14766)**
- **[GitHub Repository](https://github.com/DataDog/toto.git)**
- **[Blog Post](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/)**
- **[BOOM Dataset](https://huggingface.co/datasets/Datadog/BOOM)**

---


## 📖 Citation
If you use Toto in your research or applications, please cite us using the following:

```bibtex
@misc{cohen2025timedifferentobservabilityperspective,
      title={This Time is Different: An Observability Perspective on Time Series Foundation Models}, 
      author={Ben Cohen and Emaad Khwaja and Youssef Doubli and Salahidine Lemaachi and Chris Lettieri and Charles Masson and Hugo Miccinilli and Elise Ramé and Qiqi Ren and Afshin Rostamizadeh and Jean Ogier du Terrail and Anna-Monica Toon and Kan Wang and Stephan Xie and Zongzhe Xu and Viktoriya Zhukova and David Asker and Ameet Talwalkar and Othmane Abou-Amal},
      year={2025},
      eprint={2505.14766},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.14766}, 
}
```