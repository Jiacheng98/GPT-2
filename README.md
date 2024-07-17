# GPT-2 from Scratch on M1 Mac
This repository contains the implementation of GPT-2 built from scratch and tested on an M1 Mac.

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


### Running in Docker

You can also run the project in a Docker container by following these steps:

1. **Pull the Docker image**:

    ```bash
    docker pull robd003/python3.10:latest
    ```

2. **Build the Docker image**:

    ```bash
    sudo docker build -t gpt-2 .
    ```

3. **Run a Docker container**:

    ```bash
    sudo docker run -it --name gpt-2 -v $(pwd)/log:/app/log gpt-2 /bin/bash
    ```

This command creates a Docker container named `gpt-2` using the built Docker image `gpt-2`, mounts your local `log` directory to the container's `/app/log` directory, and opens a bash shell in the container.


### Dataset

The dataset used for training is the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The dataset is stored in the `input.txt` file.

### Training
To train the GPT-2 model from scratch using the Tiny Shakespeare dataset, run the following command:
    ```
    python main.py
    ```

### Logs
Training logs can be found in the log.txt file under the log folder. The following figure shows the training and validation loss changes with the number of steps:

![Training and validation loss](figure/loss.png)

Testing loss: 5.865208148956299


## Model Generates Samples

The following are sample outputs generated by the model given the input "The course of true love never did run smooth." with a maximum of 50 tokens:
>> The course of true love never did run smooth.

So love I should come to speak, your lord!

KING RICHARD II:
Nay, my name and love me with me crown, I come, or heart, and

>> The course of true love never did run smooth.

KING RICHARD II:
What that my sovereign-desets,
The time shall fly on him that be your love the state upon yourlook'er tears I love thou

>> The course of true love never did run smooth.

BENVOLIO:
My Lord of England's heart and thine eyes:
So far I Lord'd good hands so graces onbrokely so love'er, I

>> The course of true love never did run smooth.

RICHARD:
He would, he's a king, which did come;
The king was a men I say that heart will be with youeech' woe, I

>> The course of true love never did run smooth.

KING RICHARD II:
What say I see the king?

RATCLIFF:
Your news, my lord:
HowEN: go should lie:
My

## Reference

This project is inspired by Andrej Karpathy's tutorial. You can watch the detailed explanation in his [YouTube video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=483s&ab_channel=AndrejKarpathy).