# Repository for "InstaLearn: Continual Knowledge Updates for Domain Adaptation"

## Environment setup
```bash 
source env_setup.sh
```

## Dataset Loading
- pubmed25_cardio: [pubmed_cardio_loader.py](src/pubmed_cardio_loader.py)
- pubmed25_breast_cancer: [pubmed_breast_cancer_loader.py](/InstaLearn/src/pubmed_breast_cancer_loader.py)

## Finetune Pipeline [Instalearn Methodology]
- setup config in [](configs/llama3.2_1B_breast_cancer.yaml)

- Run finetune pipeline. Example
```bash 
python pubmed_breast_cancer_finetune_pipeline.py ../configs/qwen25_1.5B_breast_cancer.yaml
```
- checkpionts are saved in `checkpoints` folder

## Evaluation
- load model from the checkpoints folders after training.
- Our Trained models can be found in [InstaLearn/finetuned_models](https://huggingface.co/InstaLearn/finetuned_models)
- evaluation scripts can be found in [pubmedqa_eval.ipynb](pubmedqa_eval.ipynb) file. 






