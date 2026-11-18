# Skin Lesion Classification and Missing Metadata

I use PAD-UFES-20 to test whether the pattern of missing clinical fields can predict the diagnosis. I also compare two image-and-metadata models to see how randomly hiding fields during training changes their performance when information is unavailable at evaluation time.

## Data and setup

Use Python 3.11 or 3.12. Download [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1). Put `metadata.csv` at `data/metadata.csv` and extract the three image archives under `data/images/`. The PNGs may remain in their original subfolders. The loader matches each filename to `img_id` and checks that all 2,298 images are present. The verified run used the [CC BY 4.0 mirror](https://huggingface.co/datasets/SalmaneExploring/pad-ufes-20/tree/086dd94836d7b2094132a21406a75dcb6bf7ec2c) with a byte-identical metadata CSV.

On Windows or Linux, start with the CPU wheels if you do not already have a compatible PyTorch installation:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Run these commands from the repository root.

```bash
python -m notebook notebooks/01_missingness_baseline.ipynb
python -m notebook notebooks/02_model_comparison.ipynb
```

Run the cells in order. The first notebook only needs the CSV. The second notebook runs the full comparison. 

The first full run extracts 512 frozen ResNet-18 features per image and may take several minutes. The script reuses those features on later runs. Delete `data/processed/resnet18_features.npz` before rerunning if you replace the photos. The raw dataset and generated per-image predictions stay under ignored paths.

## Launch the interface

After training, run:

```bash
python -m streamlit run ui/app.py
```

Choose a recorded diagnosis and an example from the first held-out fold. Hide some of the recorded fields to compare the scores from standard and masked training. Open **Evaluation across all five folds** for the aggregate results.

## Results and limitations

The missingness-only model reached 26.9% balanced accuracy and 0.219 macro F1, compared with 16.7% and 0.090 for the majority-class baseline. It identified none of the 52 melanoma images correctly. The project uses five stratified patient-grouped folds, and reports per class recall and 95% patient bootstrap intervals from the pooled held out predictions. Image-only balanced accuracy was 53.3%, full metadata was 66.5%, and demographics alone was 45.8%. At 50% test masking, standard fusion reached 45.5% balanced accuracy and masked fusion reached 61.1%. See `results/prototype_metrics.csv` and `results/prototype_paired_differences.csv` for the full comparison.

Test masking is simulated on this dataset. It does not establish performance at another clinic or under a different documentation workflow. The interface uses existing held out examples for research.

## References 

- Pacheco et al., [PAD-UFES-20](https://doi.org/10.1016/j.dib.2020.106221), *Data in Brief*, 2020. The dataset is distributed under CC BY 4.0.
- He et al., [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90), CVPR, 2016. Image features use [TorchVision ResNet-18](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet18.html).
- Deng et al., [Multimodal transfer learning for smartphone lesion images](https://doi.org/10.1177/11769351251349891), 2025. The architecture here adapts the image-and-clinical feature concatenation to six diagnoses and a ResNet-18 backbone.
- Bouzon et al., [MetaBlock-SE](https://doi.org/10.1109/JBHI.2025.3612837), 2025, and [PRISM](https://doi.org/10.1038/s41598-026-47756-4), 2026. Both study incomplete metadata in this domain.
- Covert et al., [Explaining by Removing](https://www.jmlr.org/papers/v22/20-1316.html), 2021, discusses introducing missing features during training.

## Author

Andrew Dodds - [GitHub](https://github.com/AJDodds3)
Contact: andrew.dodds@ufl.edu / andrewdodds0715@gmail.com
