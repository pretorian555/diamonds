## we are going to preprocess our data through Transform library

## we will need to define a preprocessing module (a python function that will preprocess the data)

##the Transform component is looking for preprocessing_fn  in the module 

## steps to define preprocessing function

## 1. define list of columns based on the data they contain (numerical, cathegorical, etc )

## 2. define a preprocessing_fn

##this is a preprocessing function based on Transform tfx module

import tensorflow as tf
import tensorflow_transform as tft

import importlib

from model import features
importlib.reload(features)

##import pipeline.features as features


##create a list of feature names here

def transformed_name(key):
  return key + '_xf'

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def preprocessing_fn(inputs):

  ##create outputs out of inputs 
  outputs = {}

  ##preprocess numerical 
  for feature in features.NUMERICAL_FEATURES:
    ##we have to use dense tensors in our keras layers 
    outputs[transformed_name(feature)] = tft.scale_to_z_score(_fill_in_missing(inputs[feature]))

  for feature in features.CATEGORICAL_FEATURES:
    outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
        x = _fill_in_missing(inputs[feature]),
        num_oov_buckets=1,
        vocab_filename=feature
    )  

  ##for feature in features.LABEL_KEY:
    feature = features.LABEL_KEY
    outputs[transformed_name(feature)] = _fill_in_missing(inputs[feature])

  return outputs
