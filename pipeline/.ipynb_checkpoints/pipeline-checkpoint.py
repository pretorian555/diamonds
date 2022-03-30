
from typing import List, Optional

import tensorflow_model_analysis as tfma
import tfx
from model import features

from ml_metadata.proto import metadata_store_pb2

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.v1.proto.TrainArgs,
    eval_args: tfx.v1.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None,
) -> tfx.v1.dsl.Pipeline:


  components = []

  ##ExampleGen 
  output_config = tfx.proto.example_gen_pb2.Output(
    split_config = tfx.proto.example_gen_pb2.SplitConfig(splits=[
          tfx.proto.example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
          tfx.proto.example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)

    ])
  )

  example_gen = tfx.components.CsvExampleGen(
    input_base = data_path,
    output_config = output_config
  )

  ##adding ExampleGen component 
  components.append(example_gen)

  ## StatisticsGen
  statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples']
  )

  components.append(statistics_gen)

  ##SchemaGen

  if schema_path is None:
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'],
                       infer_feature_shape=False)
    
  else:
    schema_gen = tfx.components.SchemaGen(schema_file=schema_path)

  components.append(schema_gen)


  ##Example validator

  
  ##Transform
  transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    ##module_file='pipeline/preprocessing.py'
    preprocessing_fn=preprocessing_fn
  )

  components.append(transform)

  ##Trainer
  trainer = tfx.components.Trainer(
    ##module_file=TRAINER_MODULE,
    run_fn=run_fn,
    examples=transform.outputs['transformed_examples'],
    ##transformed_examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    transform_graph=transform.outputs['transform_graph'],
    ##train_args=trainer_pb2.TrainArgs(num_steps=3000),
    ##eval_args=trainer_pb2.EvalArgs(num_steps=1000)
    train_args=train_args,
    eval_args=eval_args
  )

  components.append(trainer)

  ##add more components as needed

  return tfx.v1.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )



