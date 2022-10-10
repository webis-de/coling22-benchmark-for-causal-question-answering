# A Benchmark for Causal Question Answering

Code and data will be finalized closer to the conference.

### Data

You can [download](https://webis.de/data.html?q=webis-causalqa-22) the Webis-CausalQA-22 corpus. To recreate the ELI5 part, check instructions bellow.

The 10 datasets used to construct Webis-CausalQA-22 corpus:

| Dataset           | Source                                  | License                                    | License type                 |
|:------------------|:----------------------------------------|:-------------------------------------------|:-----------------------------|
|`PAQ`              |https://github.com/facebookresearch/PAQ  |https://github.com/facebookresearch/PAQ#data-license |CC BY-SA 3.0         |
|`GooAQ`            |https://github.com/allenai/gooaq/        |https://github.com/allenai/gooaq/blob/main/LICENSE   |Apache License V. 2.0|
|`MS MARCO`         |https://microsoft.github.io/msmarco/     |same as source                                       |Own Terms            |
|`Natural Questions`|https://ai.google.com/research/NaturalQuestions/download |same as source                       |CC BY-SA 3.0         |
|`ELI5`             |https://github.com/facebookresearch/ELI5 or https://huggingface.co/datasets/eli5 (used)|same as source |Hosting not allowed |
|`SearchQA`         |https://github.com/nyu-dl/dl4ir-searchQA |same as source                                       |No information       |
|`SQuAD 2.0`        |https://rajpurkar.github.io/SQuAD-explorer/ |same as source                                    |CC BY-SA 4.0         |
|`NewsQA`           |https://github.com/Maluuba/newsqa        |same as source                                       |Own Terms            |
|`HotpotQA`         |https://hotpotqa.github.io/              |same as source                                       |CC BY-SA 4.0         |
|`TriviaQA`         |https://nlp.cs.washington.edu/triviaqa/index.html |same as source                              |No information       |

`ELI5` is also available in Hugging Face https://huggingface.co/datasets/eli5 that contains a script for downloading the data. This blog post provides a guide of how to download the data as well: https://yjernite.github.io/lfqa.html (was used).

*Example to obtain the `ELI5` data*

```
pip install nlp

import nlp
eli5 = nlp.load_dataset('eli5')

train_set = eli5['train_eli5']
val_set = eli5['validation_eli5']
```

Use the [regex rules](rules/causal-rules.ipynb) to identify causal questions.
