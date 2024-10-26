<div align="center">

[![PyPI version](https://badge.fury.io/py/S3.svg)](https://badge.fury.io/py/S3)
[![Arxiv](https://img.shields.io/static/v1?label=arXiv&message=2405.20082&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2405.20082)
[![NeurIPS](https://img.shields.io/static/v1?label=NeurIPS&message=Poster&color=B31B1B&logo=arXiv)](https://nips.cc/virtual/2024/poster/92935)
[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/license/apache-2-0)

</div>

<h1 align="center">Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations</h1>
<h3 align="center">
Shivam Grover
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Amin Jalali
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Ali Etemad
</h3>
<h3 align="center">
NeurIPS 2024
</h3>
<h3 align="center"> 
<a href="https://arxiv.org/pdf/2405.20082">[Paper]</a>
</h3>

## Overview

S3 is a simple plug-and-play neural network component designed to enhance time-series representation learning. S3 works by segmenting the input time-series, shuffling the segments in a learned, task-specific manner, and stitching the shuffled segments back together. S3 is modular and can be stacked to create varying degrees of granularity, integrating seamlessly with many neural architectures (e.g., CNNs, Transformers) with minimal computational overhead. It has shown improvements in both time-series classification and forecasting tasks.

## Key Features

- **Segment-Shuffle-Stitch**: Segment the input time-series, shuffle segments based on learned parameters, and stitch them back together with the original sequence for enhanced learning.
- **Easy to use**: With just 2 lines of code, you can integrate multiple S3 layers into your model.
- **Lightweight**: S3 adds minimal computational overhead to existing models.
- **Versatile**: Integrates easily with various types of models, including CNNs, Transformers, and others.

## Installation

You can install S3 via PyPI:

```bash
pip install s3-timeseries
```

Or install from the source:

```bash
git clone https://github.com/shivam-grover/S3-TimeSeries.git
cd S3
pip install .
```

## Usage
Here's how to incorporate S3 into your PyTorch model:
```python
import torch
from S3 import S3

# Sample input: batch_size = 32, time_steps = 96, features = 9
x = torch.randn(32, 96, 9)

# Initialize S3 with your desired configuration
s3_layers = S3(num_layers=3, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2)

# Apply the S3 layer
output = s3_layers(x)
```

### Arguments for `S3`

| **Parameter**          | **Explanation**                                                                                         | **Range / Sample Values**                       |
|------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `num_layers`           | Number of S3 layers to stack.                                                                            | Positive integer (e.g., 1, 2, 3)                   |
| `initial_num_segments`  | Number of segments in the first layer. This will be used only if `segments_per_layer` is not provided.   | Positive integer (e.g., 4, 8, 16)                   |
| `segment_multiplier`    | Multiplier for the number of segments in each consecutive layer.                                        | Positive float or integer (e.g., 0.5, 1, 2)        |
| `shuffle_vector_dim`    | Dimensionality of the shuffle vector, controlling shuffle complexity.                                   | Positive integer (e.g., 1, 2, 3)                |
| `use_conv_w_avg`        | Whether to use convolution-based weighted averaging.                                                    | `True`, `False`                                 |
| `initialization_type`   | **Optional**. Initialization type for shuffle vectors, such as "kaiming" or "manual".                                 | `"kaiming"`, `"manual"`                         |
| `use_stitch`            | **Optional**. Whether to use stitching to combine shuffled and original sequences.                                    | `True`, `False`                                 |
| `segments_per_layer`    | **TODO**. An array specifying the exact number of segments for each layer. Overrides `initial_num_segments` and `segment_multiplier`. | List of integers (e.g., [4, 8, 16])             |

## Citation
If you find this repository useful, please consider giving a star and citing it using the given BibTeX entry:
```bibtex
@inproceedings{
      S3TimeSeries,
      title={Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations},
      author={Shivam Grover, Amin Jalali, Ali Etemad},
      booktitle={Neural Information Processing Systems (NeurIPS)},
      year={2024},
      url={https://arxiv.org/pdf/2405.20082}
}
```

## Contact
Please contact me at <shivam.grover@queensu.ca> or connect with me on [LinkedIn](https://www.linkedin.com/in/shivam-grover/).
