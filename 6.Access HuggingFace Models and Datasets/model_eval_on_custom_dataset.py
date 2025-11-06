# Instantiate the task evaluator
task_evaluator = evaluator("image-classification")

task_evaluator.METRIC_KWARGS = {"average": "weighted"}

# Get label map from pipeline
label_map = pipe.model.config.label2id

# Compute the metrics
eval_results = task_evaluator.compute(model_or_pipeline=pipe, data=dataset, 
                         metric=evaluate.combine(metrics_dict), label_mapping=label_map)

print(f"Precision: {eval_results['precision']:.2f}, Recall: {eval_results['recall']:.2f}")