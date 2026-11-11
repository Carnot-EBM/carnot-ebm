---
task_ids:
- fact-checking
dataset_info:
- config_name: all
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: wikipedia_url
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: train_unlabeled
    num_bytes: 9216172
    num_examples: 3351
  - name: validation
    num_bytes: 1613854
    num_examples: 499
  - name: test
    num_bytes: 5352715
    num_examples: 1902
  download_size: 8621070
  dataset_size: 16182741
- config_name: ar
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 72179
    num_examples: 50
  - name: test
    num_bytes: 230978
    num_examples: 150
  download_size: 161150
  dataset_size: 303157
- config_name: ca
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: test
    num_bytes: 170272
    num_examples: 100
  download_size: 90074
  dataset_size: 170272
- config_name: cs
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: test
    num_bytes: 302156
    num_examples: 100
  download_size: 190063
  dataset_size: 302156
- config_name: de
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 76085
    num_examples: 50
  - name: test
    num_bytes: 223444
    num_examples: 150
  download_size: 188381
  dataset_size: 299529
- config_name: en
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: wikipedia_url
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: train_unlabeled
    num_bytes: 1005056
    num_examples: 809
  - name: validation
    num_bytes: 277350
    num_examples: 50
  - name: test
    num_bytes: 884853
    num_examples: 154
  download_size: 843463
  dataset_size: 2167259
- config_name: es
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: wikipedia_url
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: train_unlabeled
    num_bytes: 1475118
    num_examples: 492
  - name: validation
    num_bytes: 185035
    num_examples: 50
  - name: test
    num_bytes: 568114
    num_examples: 152
  download_size: 1247162
  dataset_size: 2228267
- config_name: eu
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: test
    num_bytes: 175533
    num_examples: 99
  download_size: 101836
  dataset_size: 175533
- config_name: fa
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: test
    num_bytes: 138131
    num_examples: 100
  download_size: 49619
  dataset_size: 138131
- config_name: fi
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 97032
    num_examples: 50
  - name: test
    num_bytes: 304309
    num_examples: 150
  download_size: 196572
  dataset_size: 401341
- config_name: fr
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: wikipedia_url
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: train_unlabeled
    num_bytes: 5857371
    num_examples: 1850
  - name: validation
    num_bytes: 170475
    num_examples: 50
  - name: test
    num_bytes: 390500
    num_examples: 150
  download_size: 3872340
  dataset_size: 6418346
- config_name: hi
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 99648
    num_examples: 50
  - name: test
    num_bytes: 265202
    num_examples: 150
  download_size: 182151
  dataset_size: 364850
- config_name: it
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_tokens
    sequence: string
  - name: model_output_logits
    sequence: float64
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 79168
    num_examples: 50
  - name: test
    num_bytes: 230517
    num_examples: 150
  download_size: 130856
  dataset_size: 309685
- config_name: sv
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: wikipedia_url
    dtype: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: validation
    num_bytes: 71091
    num_examples: 49
  - name: test
    num_bytes: 190266
    num_examples: 147
  download_size: 105372
  dataset_size: 261357
- config_name: zh
  features:
  - name: id
    dtype: string
  - name: lang
    dtype: string
  - name: model_input
    dtype: string
  - name: model_output_text
    dtype: string
  - name: model_id
    dtype: string
  - name: wikipedia_url
    dtype: string
  - name: soft_labels
    list:
    - name: start
      dtype: int64
    - name: prob
      dtype: float64
    - name: end
      dtype: int64
  - name: hard_labels
    sequence:
      sequence: int64
  - name: model_output_logits
    sequence: float64
  - name: model_output_tokens
    sequence: string
  - name: annotations
    list:
    - name: annotator_id
      dtype: string
    - name: labels
      sequence:
        sequence: int64
  splits:
  - name: train_unlabeled
    num_bytes: 878627
    num_examples: 200
  - name: validation
    num_bytes: 485791
    num_examples: 50
  - name: test
    num_bytes: 1278440
    num_examples: 150
  download_size: 1408725
  dataset_size: 2642858
configs:
- config_name: all
  data_files:
  - split: train_unlabeled
    path: all/train_unlabeled-*
  - split: validation
    path: all/validation-*
  - split: test
    path: all/test-*
- config_name: ar
  data_files:
  - split: validation
    path: ar/validation-*
  - split: test
    path: ar/test-*
- config_name: ca
  data_files:
  - split: test
    path: ca/test-*
- config_name: cs
  data_files:
  - split: test
    path: cs/test-*
- config_name: de
  data_files:
  - split: validation
    path: de/validation-*
  - split: test
    path: de/test-*
- config_name: en
  data_files:
  - split: train_unlabeled
    path: en/train_unlabeled-*
  - split: validation
    path: en/validation-*
  - split: test
    path: en/test-*
- config_name: es
  data_files:
  - split: train_unlabeled
    path: es/train_unlabeled-*
  - split: validation
    path: es/validation-*
  - split: test
    path: es/test-*
- config_name: eu
  data_files:
  - split: test
    path: eu/test-*
- config_name: fa
  data_files:
  - split: test
    path: fa/test-*
- config_name: fi
  data_files:
  - split: validation
    path: fi/validation-*
  - split: test
    path: fi/test-*
- config_name: fr
  data_files:
  - split: train_unlabeled
    path: fr/train_unlabeled-*
  - split: validation
    path: fr/validation-*
  - split: test
    path: fr/test-*
- config_name: hi
  data_files:
  - split: validation
    path: hi/validation-*
  - split: test
    path: hi/test-*
- config_name: it
  data_files:
  - split: validation
    path: it/validation-*
  - split: test
    path: it/test-*
- config_name: sv
  data_files:
  - split: validation
    path: sv/validation-*
  - split: test
    path: sv/test-*
- config_name: zh
  data_files:
  - split: train_unlabeled
    path: zh/train_unlabeled-*
  - split: validation
    path: zh/validation-*
  - split: test
    path: zh/test-*
language:
- ar
- ca
- cs
- de
- en
- es
- eu
- fa
- fi
- fr
- hi
- it
- sv
- zh
license: cc-by-4.0
task_categories:
- token-classification
pretty_name: Mu-SHROOM
---
# The **Mu-SHROOM** dataset for Multilingual Hallucination and Overgeneration detection.
Mu-SHROOM: Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes and Related Observable Overgeneration Mistakes

## Dataset Description

Mu-SHROOM is a multilingual dataset for detecting hallucination spans in LLM outputs across 14 languages. It was created for [SemEval-2025 Task 3](https://helsinki-nlp.github.io/shroom/2025).

**disclaimer**: Mu-SHROOM is not properly a fact-checking dataset, but we mark is as such until `hallucination detection` (or something more adequate) is added to the offical listo of task_ids.
### Features
- **14 languages**: Arabic, Basque, Catalan, Chinese, Czech, English, Farsi, Finnish, French, German, Hindi, Italian, Spanish, Swedish
- **Splits**: `train_unlabeled`, `validation`, and `test` sets
- **Rich annotations**: Character-level hallucination spans with hard and soft labels, and annotator IDs
- **Model outputs**: Includes output tokens and logits from various LLMs
- **Full Transparency**: For full replicability, in [the official git repo](https://github.com/Helsinki-NLP/mu-shroom) we make available all the scripts
                         used to generate the outputs. We also grant access to scripts to replicate the annotation & evaluation pratforms, evaluation
                          scripts, the raw data and the shared-task participant kit.

## Dataset Structure

Each language is available as a separate subset, with "all"-.subset contianing a concatenation of all data. 
The dataset contains:

### Data Fields
- `id`: Unique example identifier
- `lang`: Language code (ISO 639-1)
- `model_input`: The input prompt given to the LLM
- `model_output_text`: The generated output text
- `model_id`: Identifier of the LLM that generated the output
- `wikipedia_url`: Reference Wikipedia URL used for annotation
- `soft_labels`: Probabilistic character spans of hallucinations `[{"start": int, "end": int, "prob": float}]`
- `hard_labels`: Binary character spans of hallucinations `[[start, end]]` (marked as 1 when the majority of annotators marked is as a hallucination)
- `model_output_logits`: Logits from the LLM generation
- `model_output_tokens`: Tokenized output
- `annotations`: Raw annotations from multiple annotators `[{"annotator_id": str, "labels": [[start, end]]}]`
  - `annotator_id`: unique identifier for each annotator (useful for studying annotation trends, like disagreement)

### Data Splits
Each language has:
- `train_unlabeled`: Unlabeled training data (available for some languages)
- `validation`: Labeled validation set
- `test`: Labeled test set

## How to Use

### Loading the Dataset

```python
from datasets import load_dataset

# Load a specific language (e.g., English)
dataset = load_dataset("Helsinki-NLP/mu-shroom", "en")

# Access splits
train = dataset["train_unlabeled"]
val = dataset["validation"]
test = dataset["test"]
```
### Load all languages combined
```python
full_dataset = load_dataset("Helsinki-NLP/mu-shroom", "all")
```

### Example Usage
```python
# Get an example from validation set
example = dataset["validation"][0]

print(f"Language: {example['lang']}")
print(f"Input: {example['model_input']}")
print(f"Model Output: {example['model_output_text']}")
print(f"Hallucination spans: {example['hard_labels']}")

# Visualize hallucination spans
text = example["model_output_text"]
for span in example["hard_labels"]:
    start, end = span
    print(f"Hallucinated text: '{text[start:end]}'")
```
Expected outcome
```text
Language: en
Input: What did Petra van Staveren win a gold medal for?
Model Output: Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.
Hallucination spans: [[25, 31], [45, 49], [69, 83]]
Hallucinated text: 'silver'
Hallucinated text: '2008'
Hallucinated text: 'Beijing, China'
```

# Shared Task Information: Quick Overview
Mu-SHROOM is part of SemEval-2025 Task 3. Participants were asked to detect hallucination spans in LLM outputs.
They were evaluatied using [this evaluation script](https://github.com/Helsinki-NLP/mu-shroom/blob/main/participant_kit/scorer.py) over the:
- intersection-over-Union (IoU) of hallucinated characters
- correlation between predicted and empirical probabilities



# Citation
If you use this dataset, please cite the SemEval-2025 task proceedings *(citation information to be updated after the workshop)*:
```bib
@inproceedings{vazquez-etal-2025-mu-shroom,
  author={Ra\'ul V\'azquez and Timothee Mickus and Elaine Zosa and Teemu Vahtola and J\"org Tiedemann and Aman Sinha and Vincent Segonne and Fernando S\'anchez-Vega and Alessandro Raganato and Jindřich Libovický and Jussi Karlgren and Shaoxiong Ji and Jindřich Helcl and Liane Guillou and Ona de Gibert and Jaione Bengoetxea and Joseph Attieh and Marianna Apidianaki},
  title={Sem{E}val-2025 {T}ask 3: {Mu-SHROOM}, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes},
  year={2025},
  url={https://helsinki-nlp.github.io/shroom/2025},
  booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
  publisher = "Association for Computational Linguistics",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
}
```

## Contact
For questions about the dataset, please contact the organizers:
- Raúl Vázquez (University of Helsinki)
- Timothee Mickus (University of Helsinki)

## 👥🙌🌐 Join the SHROOM Community 

Whether you're interested in joining the next round, learning from past editions, or just staying informed about hallucination detection in NLG, we'd love to have you in the community.
- Check out the [**\*SHRO0M** shared task series](https://helsinki-nlp.github.io/shroom/)
- Join the conversation on [Slack](https://join.slack.com/t/shroom-shared-task/shared_invite/zt-2mmn4i8h2-HvRBdK5f4550YHydj5lpnA)
- Check out the past editions Google gorups
  - [Mu-SHROOM 2025](https://groups.google.com/g/semeval-2025-task-3-mu-shroom)
  - [Mu-SHROOM 2024](https://groups.google.com/g/semeval-2024-task-6-shroom)