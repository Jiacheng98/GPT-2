# GPT-2 from Scratch on M1 Mac
This repository contains the implementation of GPT-2 built from scratch and tested on an M1 Mac.

## Dataset

The dataset used for training is the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The dataset is stored in the `input.txt` file.

## Reference

This project is inspired by Andrej Karpathy's tutorial. You can watch the detailed explanation in his [YouTube video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=483s&ab_channel=AndrejKarpathy).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Other dependencies as listed in `requirements.txt`

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Training
To train the GPT-2 model from scratch using the Tiny Shakespeare dataset, run the following command:
    ```bash
    python main.py
    ```

### Logs
Training logs can be found in the log.txt file under the log folder.

