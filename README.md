# [NeurIPS 2025] ATLAS

📝 Official implementation for the paper:

[ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](https://arxiv.org/abs/2502.05567)


## Dataset and Model Downloads
The ATLAS Translator series models and the ATLAS dataset used in this paper are publicly available at the following repository: [🤗 HuggingFace](https://huggingface.co/collections/XiaoyangLiu-sjtu/atlas-68d63dffd479f2553b1ca7d9).


## Project Structure
The repository is organized as follows.

```text
ATLAS/
├── benchmarks/                         # Benchmark datasets in jsonl format
│   ├── minif2f.jsonl
│   ├── proofnet.jsonl
│   ├── putnam.jsonl
│   └── mathqual.jsonl
├── configs/                            # Model settings and prompts
│   ├── concept_repository.json         # Concept pool used in data generation
│   ├── config_generation.py            # Configs for ATLAS data generation pipeline
│   └── config_evaluation.py            # Configs for benchmark evaluation pipeline
├── src/                                # Core implementation
│   ├── repl/                           # Lean4 REPL project used for compilation checks
│   │   ├── REPL/
│   │   ├── REPL.lean
│   │   ├── lakefile.toml
│   │   └── lean-toolchain
│   └── workers/                        # Modular workers for generation/evaluation
│       ├── synthesizer.py              # NL/FL generation, revision, alignment, translation models
│       ├── augmenter.py                # Formal statement parsing and augmentation
│       ├── verifier.py                 # Lean verification process scheduler/worker pool
│       ├── back_translator.py          # Back-translation worker for semantic checking
│       └── nli_checker.py              # NLI-based semantic consistency checker
├── generation.py                       # End-to-end iterative ATLAS data generation script
├── evaluation.py                       # Benchmark evaluation script (compile + semantic checks)
├── utils.py                            # Shared utilities: model wrappers, I/O, verification, summaries
├── LICENSE
└── README.md
```


## Quick Start
1. **Install Lean4.** Follow the official [Lean4 installation guide](https://leanprover-community.github.io/get_started.html).
2. **Clone the repository.** Clone this repository and enter the project directory.
3. **Build the project.** Follow the instructions on the [Lean REPL page](https://github.com/leanprover-community/repl.git) to set up Lean REPL. Then install Python dependencies, build the local REPL project in `src/repl`, and update model/API settings in `configs/config_generation.py` and `configs/config_evaluation.py`.
4. **Evaluation.** There are two common entry points for generation and evaluation.
    - `generation.py`: Runs the full ATLAS iterative data generation pipeline (NL generation/translation, revision, alignment, augmentation).
        ```shell
        # Entry 1: generation
        python generation.py
        ```

    - `evaluation.py`: Runs benchmark evaluation (translation, Lean compile check, back-translation, NLI check). You can modify `models`, `seeds`, `benchmarks`, and `number_to_generates` in `evaluation.py`.
        ```shell
        # Entry 2: evaluation
        python evaluation.py
        ```
    

## Citation
```bibtex
@inproceedings{liu2026atlas,
title={{ATLAS}: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data},
author={Xiaoyang Liu and Kangjie Bao and Jiashuo Zhang and Yunqi Liu and Yu Chen and Yuntian Liu and Yang Jiao and Tao Luo},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=MlJyAvQaxp}
}
```


## Contact
Feel free to discuss the paper/data/code with us through issues/emails!
- Xiaoyang Liu: xiaoyang.liu@sjtu.edu.cn


## Acknowledgement
This repo benefits from [Herald Translator](https://github.com/frenzymath/herald_translator) and [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5). Thanks for their wonderful works.
