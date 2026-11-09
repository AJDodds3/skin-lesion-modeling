# Skin Lesion Classification and Missing Metadata

This project looks at whether missing clinical fields cause skin lesion classifiers to learn misleading patterns. I plan to compare images, clinical metadata, and a combination of both, then test whether randomly hiding metadata during training makes the model more reliable.

## Setup

Install Python and run:

```bash
python -m pip install -r requirements.txt
python -m notebook setup.ipynb
```

Run both cells in the notebook. They load the dataset, show the number of cases in each diagnosis, and count missing fields by diagnosis.

## Dataset

I am using [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1), which has 2,298 skin lesion images and clinical metadata for six diagnoses.

Download the `metadata.csv` file from the dataset and put it at `data/metadata.csv`. The notebook will use the full file if it is present. Until then, it uses `data/sample_metadata.csv` just to check that the environment works. **The six-row sample is not enough for real project results.**

The original data is released under CC BY 4.0. Please cite Pacheco et al. (2020), *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*, Data in Brief, 32, 106221.

## Author

Andrew Dodds — [GitHub](https://github.com/AJDodds3)
