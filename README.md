# SwissJudgementPrediction
In many jurisdictions, the excessive workload of courts leads to high delays. Suitable predictive AI models can assist legal professionals in their work, and thus enhance and speed up the process. So far, Legal Judgment Prediction (LJP) datasets have been released in English, French, and Chinese. We publicly release a multilingual (German, French, and Italian), diachronic (2000-2020) corpus of 85K cases from the Federal Supreme Court of Switzerland (FSCS). We evaluate state-of-the-art BERT-based methods including two variants of BERT that overcome the BERT input (text) length limitation (up to 512 tokens). Hierarchical BERT has the best performance (approx. 68-70% Macro-F1-Score in German and French). Furthermore, we study how several factors (canton of origin, year of publication, text length, legal area) affect performance. We release both the benchmark dataset and our code to accelerate future research and ensure reproducibility.

This repository provides code for experiments with the state-of-the-art in text classification to predict the judgements of Swiss court decisions

## Dataset
The data is available on Zenodo (https://zenodo.org/record/5529712) and HuggingFace Datasets (http://huggingface.co/datasets/swiss_judgment_prediction). 

## Paper
ArXiv pre-print is available here: http://arxiv.org/abs/2110.00806
You can cite it as follows: 

    @misc{niklaus2021swissjudgmentprediction,
        title={Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark},
        author={Joel Niklaus and Ilias Chalkidis and Matthias Stürmer},
        year={2021},
        eprint={2110.00806},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
