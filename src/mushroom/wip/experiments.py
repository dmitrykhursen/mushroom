#%%
from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.pipeline import Pipeline
from pathlib import Path


#%%
project_root = Path(settings.project_root)
print(f"Project root: {project_root}")

def run_experiment(
    name: str,
    original_file_path_relative: str,
    steps: list,
    config = None,
    step_file_path_relative: str = None,
):
    config = config if config is not None else settings
    
    file_path = project_root / original_file_path_relative 
    print(f"File path: {file_path}")
    dataset_name = Path(file_path).stem
    print(f"Dataset name: {dataset_name}")
    
    if step_file_path_relative is not None:
        file_path = project_root / step_file_path_relative
        print(f"File path from the last stage: {file_path}")
    
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    

    output_path = outputs_dir / f"{dataset_name}_{name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file_path = output_path / "predictions.jsonl"
    # if output_file_path.exists():
    #     print(f"Output file already exists: {output_file_path}")
    #     return
    
    
    pipeline = Pipeline(config=config, steps=steps)
    print(f"Pipeline: {pipeline.steps}")
    predictions = pipeline(file_path, output_file_path)
    print(f"Predictions generated")
    results = pipeline.evaluate(predictions)
    print(f"Results: {results}")
    
    
# run_experiment(
#     name="test_baseline",
#     file_path="data/extra/splits/test_labeled/v1/mushroom.en-tst.v1.extra.jsonl",
#     steps=["fact_extraction", "retrieval", "span_labeling_baseline"]
# )
# run_experiment(
#     name="test_baseline",
#     original_file_path_relative="data/extra/splits/test_labeled/v1/mushroom.en-tst.v1.extra.jsonl",
#     step_file_path_relative="outputs/mushroom.en-tst.v1.extra_test_baseline/predictions_fact_extraction.jsonl",
#     steps=["retrieval", "span_labeling_baseline"]
# )
# run_experiment(
#     name="test_main",
#     original_file_path_relative="data/extra/splits/test_labeled/v1/mushroom.en-tst.v1.extra.jsonl",
#     step_file_path_relative="outputs/mushroom.en-tst.v1.extra_test_baseline/predictions_retrieval.jsonl",
#     steps=["span_labeling"]
# )

# run_experiment(
#     name="test_main",
#     file_path="data/extra/splits/test_labeled/v1/mushroom.en-tst.v1.extra.jsonl",
#     steps=["span_labeling"]
# )

run_experiment(
    name="val_main",
    original_file_path_relative= "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl",
    step_file_path_relative="outputs/mushroom.en-val.v2.extra_val_baseline/predictions_retrieval.jsonl",
    steps=["span_labeling"]
)

# run_experiment(
#     name="val_main",
#     file_path = "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl",
#     steps=["span_labeling"]
# )    
