### Running the format checker

Assuming (i) you downloaded the unlabeled test data and unzipped it into a directory `PATH/TO/TEST`, (ii) you want to check the prediction file `file_to_check.jsonl`:

```bash
python format_checker.py PATH/TO/TEST file_to_check.jsonl
```

### Using the baseline model script

To train a model using the validation dataset excluding one test language:

```bash
python baseline_model.py --mode train --output_path ./results --data_path ../val --test_lang es
```

To get predictions from a trained model:

```bash
python baseline_model.py --mode test --model_checkpoint results/checkpoint-20/ --data_path ../val/ --test_lang es
```

Note: The soft label and hard label predictions will be written to JSON files.

