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
