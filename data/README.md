# data/

Datasets are **not** committed (`data/raw/` is gitignored). Download HC18 from Zenodo
(doi:10.5281/zenodo.1327317) and extract it as:

```text
data/raw/HC18/
├── training_set/                       # 999 *HC.png + *HC_Annotation.png
├── test_set/                           # 335 images, no public labels
├── training_set_pixel_size_and_HC.csv
└── test_set_pixel_size.csv
```

Check it with `python scripts/check_hc18.py`.

FETAL_PLANES_DB (standard-plane gate, Feature A): Zenodo record 3904280 (doi:10.5281/zenodo.3904280),
file `FETAL_PLANES_ZENODO.zip` (2,088,522,169 bytes). Extract it as:

```text
data/raw/FETAL_PLANES_DB/
├── FETAL_PLANES_DB_data.csv            # ';'-separated: Image_name, Patient_num, Plane, Brain_plane, ..., Train
├── FETAL_PLANES_DB_data.xlsx
└── Images/                             # 12,400 PNG images
```

`python -m ml.train_planes` caches letterboxed images in `data/processed/` (gitignored) and uses the committed
split `data/splits/fetal_planes_split.csv`.
