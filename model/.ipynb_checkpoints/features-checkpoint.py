
NUMERICAL_FEATURES = ['carat','depth','table','x','y','z']

CATEGORICAL_FEATURES = ['clarity','color','cut']

LABEL_KEY = 'price'


def transformed_name(key):
  return key + '_xf'
