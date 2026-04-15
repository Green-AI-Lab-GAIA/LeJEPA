# LeJEPA tests using CIFAR10 and RESNET

To run the code, first install the dependencies and then execute the main script:

```bash
pip install -r requirements.txt
python lejepa.py
```

It will ask for a run name, which will be used to save the model checkpoints and logs.
The configuration parameters can be modified in the `config.yaml` file.

Python Files:
- `evaluate.py`: Contains the evaluation function to compute the accuracy of the model.
- `lejepa.py`: Main training script of the LeJEPA model.
- `transforms.py`: Contains the data augmentation and transformation functions used during training.
- `utils.py`: Utility functions for saving the logs and average meters.

Jupyter Notebooks:
- `Transforms.ipynb`: Jupyter notebook for visualizing the data transformations applied to the CIFAR10 dataset.
- `TSNE.ipynb`: Jupyter notebook for visualizing the learned embeddings using t-SNE and a confusion matrix.