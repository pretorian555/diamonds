
"""Define LocalDagRunner to run the pipeline locally."""

import os
from absl import logging

from tfx import v1 as tfx
from pipeline import configs
from pipeline import pipeline

OUTPUT_DIR='.'

PIPELINE_BUCKET = 'gs://diamonds_pipeline'

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME,
                             'metadata.db')
SERVING_MODEL_DIR = os.path.join(PIPELINE_BUCKET, 'serving_model')


DATA_PATH = 'gs://diamonds_data'

if not DATA_PATH:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def run():
  """Define a pipeline."""

  ##debug code
  print("serving model dir: ", SERVING_MODEL_DIR)
  print("metadata path: ", METADATA_PATH)

  tfx.orchestration.LocalDagRunner().run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          # NOTE: Use `query` instead of `data_path` to use BigQueryExampleGen.
          # query=configs.BIG_QUERY_QUERY,
          # NOTE: Set the path of the customized schema if any.
          # schema_path=generated_schema_path,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          tuner_fn=configs.TUNER_FN,
          enable_tuning=False,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
          # NOTE: Provide GCP configs to use BigQuery with Beam DirectRunner.
          # beam_pipeline_args=configs.
          # BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
          metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH))
      )
  
if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
