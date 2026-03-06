# PawPularity

A picture is worth a thousand words. But did you know a picture can save a thousand lives? Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. You might expect pets with attractive photos to generate more interest and be adopted faster. But what makes a good picture? With the help of data science, you may be able to accurately determine a pet photo’s appeal and even suggest improvements to give these rescue animals a higher chance of loving homes.

PetFinder.my is Malaysia’s leading animal welfare platform, featuring over 180,000 animals with 54,000 happily adopted. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Currently, PetFinder.my uses a basic Cuteness Meter to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved.

In this competition, you’ll analyze raw images and metadata to predict the “Pawpularity” of pet photos. You'll train and test your model on PetFinder.my's thousands of pet profiles. Winning versions will offer accurate recommendations that will improve animal welfare.

If successful, your solution will be adapted into AI tools that will guide shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements. As a result, stray dogs and cats can find their "furever" homes much faster. With a little assistance from the Kaggle community, many precious lives could be saved and more happy families created.

Top participants may be invited to collaborate on implementing their solutions and creatively improve global animal welfare with their AI skills.

# Quickstart

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

myproj preprocess --config configs/default.yaml
myproj train --config configs/train.yaml
myproj eval --config configs/eval.yaml

```

```bash

├── README.md
├── pyproject.toml              # deps + tooling (preferred) or requirements.txt
├── .gitignore
├── .env.example                # secrets template (never commit real .env)
├── LICENSE
├── data/
│   ├── raw/                    # never commit large raw data (gitignored)
│   ├── interim/
│   └── processed/
├── notebooks/                  # exploration only; keep minimal
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── config.py           # dataclasses / pydantic config
│       ├── data/
│       │   ├── ingest.py
│       │   ├── preprocess.py
│       │   └── dataset.py
│       ├── models/
│       │   ├── model.py
│       │   ├── train.py
│       │   ├── eval.py
│       │   └── predict.py
│       ├── features/
│       │   └── build_features.py
│       ├── utils/
│       │   ├── seed.py
│       │   ├── logging.py
│       │   └── paths.py
│       └── cli.py              # command line entrypoint
├── scripts/                    # small runnable scripts (optional)
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   └── eval.yaml
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_train_smoke.py
├── artifacts/                  # model outputs (gitignored)
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
├── docs/                       # notes, experiment logs, paper
└── .github/
    └── workflows/
        └── ci.yml

```

# Reproduce My Score
