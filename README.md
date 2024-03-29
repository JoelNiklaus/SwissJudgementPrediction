# SwissJudgmentPrediction
In many jurisdictions, the excessive workload of courts leads to high delays. Suitable predictive AI models can assist legal professionals in their work, and thus enhance and speed up the process. So far, Legal Judgment Prediction (LJP) datasets have been released in English, French, and Chinese. We publicly release a multilingual (German, French, and Italian), diachronic (2000-2020) corpus of 85K cases from the Federal Supreme Court of Switzerland (FSCS). We evaluate state-of-the-art BERT-based methods including two variants of BERT that overcome the BERT input (text) length limitation (up to 512 tokens). Hierarchical BERT has the best performance (approx. 68-70% Macro-F1-Score in German and French). Furthermore, we study how several factors (canton of origin, year of publication, text length, legal area) affect performance. We release both the benchmark dataset and our code to accelerate future research and ensure reproducibility.

This repository provides code for experiments with the state-of-the-art in text classification to predict the judgements of Swiss court decisions.

## Get Started
### In UBELIX
* Read the [documentation](https://hpc-unibe-ch.github.io/quick-start.html) of UBELIX
* Open the .bashrc file in your $HOME Folder and enter `module load CUDA`
* Enter `module load Anaconda3` in the terminal
* Enter the conda environment using `eval "$(conda shell.bash hook)"`

_UBELIX is a centOS based high-performance computing cluster_

### In Conda:
* Create a new environment called "sjp" and install packages from the env.yml file using `conda env create -f env.yml`
* Activate the sjp environment using `conda avtivate sjp`.
* Use the following command to install the right version of PyTorch: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` 
  * _(If your not working on UBELIX replace "cu113" with your CUDA version. You can find it by entering `nvcc --version` in the terminal)_
* Create a second environment called "data_aug" and install the required packages following the imports of the "translator.py" file
* Create a Weights & Biases account, get your api token, and enter `wandb login` inside your conda environment. After you entered the token, it will be saved in the .netrc file in you $HOME folder


## Dataset
The data is available on Zenodo (https://zenodo.org/record/5529712) and HuggingFace Datasets (http://huggingface.co/datasets/swiss_judgment_prediction). 

## Paper
ArXiv pre-prints are available here: http://arxiv.org/abs/2110.00806, https://arxiv.org/abs/2209.12325.
You can cite them as follows: 

    @misc{niklaus2021swissjudgmentprediction,
        title={Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark},
        author={Joel Niklaus and Ilias Chalkidis and Matthias Stürmer},
        year={2021},
        eprint={2110.00806},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }



    @misc{niklaus2022empirical,
        title={An Empirical Study on Cross-X Transfer for Legal Judgment Prediction},
        author={Joel Niklaus and Matthias Stürmer and Ilias Chalkidis},
        year={2022},
        eprint={2209.12325},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
