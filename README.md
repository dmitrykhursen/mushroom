# mushroom

Project on SemEval-2025 Task-3 - Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes

## Installation

### Install uv
```
curl -Ls https://astral.sh/uv/install.sh | sh
```
### Install dependencies
```
uv sync
```


## Execution

### Download data
```
uv run src/mushroom/utils/download_data.py
```

### Run pipeline
```
uv run src/mushroom/pipeline/pipeline.py
```


## Project Structure

- [src/](./src)
  - [mushroom/](./src/mushroom)
    - [config/](./src/mushroom/config)
      - [config.py](./src/mushroom/config/config.py) - Configuration settings using Pydantic
    - [participant_kit/](./src/mushroom/participant_kit)
    - [pipeline/](./src/mushroom/pipeline)
      - [data_connector.py](./src/mushroom/pipeline/data_connector.py) - Data reading and writing utilities
      - [interface.py](./src/mushroom/pipeline/interface.py) - Entry interface
      - [pipeline.py](./src/mushroom/pipeline/pipeline.py) - Main pipeline implementation
      - [fact_extractor/](./src/mushroom/pipeline/fact_extractor)
      - [retrieval/](./src/mushroom/pipeline/retrieval)
      - [span_labeling/](./src/mushroom/pipeline/span_labeling)
        - [evaluate_span_labeling.py](./src/mushroom/pipeline/span_labeling/evaluate_span_labeling.py) - Evaluation script for span labeling
        - [span_labeling.py](./src/mushroom/pipeline/span_labeling/span_labeling.py) - Span labeling implementation
    - [utils/](./src/mushroom/utils)
      - [download_data.py](./src/mushroom/utils/download_data.py) - Script to download datasets

