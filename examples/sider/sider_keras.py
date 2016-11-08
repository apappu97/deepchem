"""
Script that trains Keras multitask models on SIDER dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc
from sider_datasets import load_sider

# Set some global variables up top
np.random.seed(123)
reload = True
verbosity = "high"
model = "logistic"


sider_tasks, dataset, transformers = load_sider()
print("len(dataset)")
print(len(dataset))

base_dir = "/tmp/sider_analysis"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Load SIDER data
sider_tasks, sider_datasets, transformers = load_sider()
train_dataset, valid_dataset, test_dataset = sider_datasets
n_features = 1024 


# Build model
classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

learning_rates = [0.0003, 0.001, 0.003]
hidden_units = [1000, 500]
dropouts = [.5, .25]
num_hidden_layers = [1, 2]

  # hyperparameter sweep here
for learning_rate in learning_rates:
  for hidden_unit in hidden_units:
    for dropout in dropouts:
      for n_layer in num_hidden_layers:
      	keras_model = dc.models.keras_models.fcnet.MultiTaskDNN(len(sider_tasks), n_features, "classification",
                                 n_layers = n_layer, dropout=.25, learning_rate=.001, decay=1e-4)
        model = dc.models.keras_models.KerasModel(keras_model, verbosity = verbosity)

        # Fit trained model
        model.fit(train_dataset)
        model.save()

        train_scores = model.evaluate(train_dataset,[classification_metric], transformers)

        print("Train scores")
        print(train_scores)

        valid_scores = model.evaluate(valid_dataset,[classification_metric], transformers)


        print("Validation scores")
        print(valid_scores)	
        with open('./results.csv', 'a') as f:
      	  f.write('learning rate, ' + str(learning_rate) + '\n')
	  f.write('hidden unit, ' + str(hidden_unit) + '\n')
	  f.write('dropout, ' + str(dropout) + '\n')
	  f.write('n_layers, ' + str(n_layer) + '\n')
	  f.write('train score, ' + str(train_scores) + '\n')
	  f.write('valid score, ' + str(valid_scores) + '\n\n')
