"""
Code for processing datasets using scikit-learn.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsCV
from deepchem.models import Model

class SklearnModel(Model):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    super(SklearnModel, self).__init__(task_types, model_params,
                                       initialize_raw_model)
    self.task_types = task_types
    self.model_params = model_params
    if initialize_raw_model:
      if self.modeltype == "rf_regressor":
        raw_model = RandomForestRegressor(
            n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
      elif self.modeltype == "rf_classifier":
        raw_model = RandomForestClassifier(
            n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
      elif modeltype == "logistic":
        raw_model = LogisticRegression(class_weight="auto")
      elif modeltype == "linear":
        raw_model = LinearRegression(normalize=True)
      elif modeltype == "ridge":
        raw_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], normalize=True)
      elif modeltype == "lasso":
        raw_model = LassoCV(max_iter=2000, n_jobs=-1)
      elif modeltype == "lasso_lars":
        raw_model = LassoLarsCV(max_iter=2000, n_jobs=-1)
      elif modeltype == "elastic_net":
        raw_model = ElasticNetCV(max_iter=2000, n_jobs=-1)
      else:
        raise ValueError("Invalid model type provided.")

  # TODO(rbharath): This does not work with very large datasets! sklearn does
  # support partial_fit, but only for some models. Might make sense to make
  # PartialSklearnModel subclass at some point to support large data models.
  def fit(self, numpy_dataset):
    """
    Fits SKLearn model to data.
    """
    Xs, ys = [], []
    for (X, y, _, _) in numpy_dataset.itershards():
      Xs.append(X)
      ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    self.raw_model.fit(X, y)

  def predict_on_batch(self, X):
    """
    Makes predictions on given batch of new data.
    """
    return self.raw_model.predict(X)

  def save(self, out_dir):
    """Saves sklearn model to disk using joblib."""
    super(SklearnModel, self).save(out_dir)
    joblib.dump(self.raw_model, self.get_model_filename(out_dir))

  def load(self, model_dir):
    """Loads sklearn model from joblib file on disk."""
    self.raw_model = joblib.load(Model.get_model_filename(model_dir)

Model.register_model_type("logistic", SklearnModel)
Model.register_model_type("rf_classifier", SklearnModel)
Model.register_model_type("rf_regressor", SklearnModel)
Model.register_model_type("linear", SklearnModel)
Model.register_model_type("ridge", SklearnModel)
Model.register_model_type("lasso", SklearnModel)
Model.register_model_type("lasso_lars", SklearnModel)
Model.register_model_type("elastic_net", SklearnModel)
