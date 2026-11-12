# Dataset preparation guide

Each manifest row must represent a genuinely paired ultrasound image and ECG recording from the same study/participant, with a verified ultrasound view label.

Required columns:
- `image_path`: path to an image file.
- `ecg_path`: path to a CSV containing at least one numeric signal column.
- `label`: one of Abdomen, Brain, Femur, Thorax, Cervix, Other.
- `patient_id`: de-identified participant/group ID.

Optional:
- `split`: train, val, or test. If supplied, all three are required and each patient must occur in exactly one split.

Do not pair unrelated images and ECG recordings. Do not include direct identifiers. Confirm data permissions, ethics approval, dataset licence, and label definitions with your supervisor.
