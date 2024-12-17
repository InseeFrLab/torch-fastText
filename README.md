# torchFastText : Efficient text classification with PyTorch

A flexible PyTorch implementation of FastText for text classification with support for categorical features.

## Features

- Supports text classification with FastText architecture
- Handles both text and categorical features
- N-gram tokenization
- Flexible optimizer and scheduler options
- GPU and CPU support
- Model checkpointing and early stopping
- Prediction and model explanation capabilities

## Installation

```bash
pip install torchFastText
```

## Key Components

- `build()`: Constructs the FastText model architecture
- `train()`: Trains the model with built-in callbacks and logging
- `predict()`: Generates class predictions
- `predict_and_explain()`: Provides predictions with feature attributions

## Subpackages

- `preprocess`: To preprocess text input, using `nltk` and `unidecode` libraries.
- `explainability`: Simple methods to visualize feature attributions at word and letter levels, using `captum`library.

Run `pip install torchFastText[preprocess]` or `pip install torchFastText[explainability]` to download these optional dependencies.


## Quick Start

```python
from torchFastText import torchFastText

# Initialize the model
model = torchFastText(
    num_buckets=1000000,
    embedding_dim=100,
    min_count=5,
    min_n=3,
    max_n=6,
    len_word_ngrams=True,
    sparse=True
)

# Train the model
model.train(
    X_train=train_data,
    y_train=train_labels,
    X_val=val_data,
    y_val=val_labels,
    num_epochs=10,
    batch_size=64
)
# Make predictions
predictions = model.predict(test_data)
```

where ```train_data``` is an array of size $(N,d)$, having the text in string format in the first column, the other columns containing tokenized categorical variables in `int` format.

Please make sure `y_train` contains at least one time each possible label.

## Dependencies

- PyTorch Lightning
- NumPy

## Documentation

For detailed usage and examples, please refer to the [experiments notebook](experiments.ipynb). Use `pip install -r requirements.txt` after cloning the repository to install the necessary dependencies (some are specific to the notebook).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
## References

Inspired by the original FastText paper [1] and implementation.

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@InProceedings{joulin2017bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={April},
  year={2017},
  publisher={Association for Computational Linguistics},
  pages={427--431},
}
```
