
from typing import List, Optional

import tensorflow_model_analysis as tfma
import tfx
from model import features

from ml_metadata.proto import metadata_store_pb2

##local variables for modules

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    preprocessing_fn: str,
    run_fn: str,
    tuner_fn: str,
    enable_tuning: bool,
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
    
    if enable_tuning:
        tuner = tfx.components.Tuner(
            ##module_file=TRAINER_MODULE,
            tuner_fn=tuner_fn,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=train_args,
            eval_args=eval_args
        )
        components.append(tuner)
        
    ##if tuner is not enabled = we already got  best parameters
    ## import best parameters from the previous session
    
    ##TODO - figure out if there is a way to import the best parameters from the MD store
    if not enable_tuning:
        hparams_importer = tfx.v1.dsl.Importer(
            source_uri = './tfx_pipeline_output/diamonds-pipeline/Tuner/best_hyperparameters/6',
            artifact_type = tfx.v1.types.standard_artifacts.HyperParameters).with_id('import_hparams')
        components.append(hparams_importer)

    ##Trainer
    trainer = tfx.components.Trainer(
        ##module_file=TRAINER_MODULE,
        run_fn=run_fn,
        examples=transform.outputs['transformed_examples'],
        ##transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        hyperparameters=(tuner.outputs['best_hyperparameters'] if enable_tuning else hparams_importer.outputs['result']),  
        ##train_args=trainer_pb2.TrainArgs(num_steps=3000),
        ##eval_args=trainer_pb2.EvalArgs(num_steps=1000)
        train_args=train_args,
        eval_args=eval_args
    )

    components.append(trainer)

    ##add more components as needed
  
    ##Evaluator
    ## evaluates the new model
    ##validates it against a base model if it is good enough to be pushed
 
    metrics_specs= tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name='ExampleCount'),
                    tfma.MetricConfig(class_name='MeanSquaredError',
                                     threshold=tfma.MetricThreshold(
                                         change_threshold=tfma.GenericChangeThreshold(
                                         absolute={'value': -1e-10}, direction=tfma.MetricDirection.LOWER_IS_BETTER)))
                ],
    ##you can add threshold map for metrics used in the model training
        thresholds = {
            'root_mean_squared_error': tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(),
                    ##you dont have to set an upper bound - will default to infinity
                    ##upper_bound={'value': 2000}),
                change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.LOWER_IS_BETTER,
                        absolute={'value': 1e-10}
                )
            )
        }

    )


    eval_config = tfma.EvalConfig(
        model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
            tfma.ModelSpec(
                signature_name='serving_default',
                label_key='price'
            ##preprocessing_function_names=['transform_features'],
                )
        ],
        metrics_specs=[metrics_specs],
        slicing_specs=[
            ##this is for the whole dataset, no slices
            tfma.SlicingSpec(),
            ##also specifying slices for some features 
            tfma.SlicingSpec(feature_keys=['cut']),
            tfma.SlicingSpec(feature_keys=['color']),
        ]
    )
    
    ##resolve the previous blessed model
    model_resolver=tfx.v1.dsl.Resolver(
        strategy_class=tfx.v1.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.v1.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.v1.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing)).with_id('latest_blessed_model_resolver')

    components.append(model_resolver)

    model_analyzer = tfx.components.Evaluator(
        examples = example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    
    components.append(model_analyzer)
    
    ##Pusher
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=model_analyzer.outputs['blessing'],
        push_destination=tfx.v1.proto.PushDestination(
            filesystem=tfx.v1.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)
        )
    )
    
    components.append(pusher)

    return tfx.v1.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
    )



