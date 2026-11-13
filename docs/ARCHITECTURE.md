# Architecture

FetalAI first checks that a 2-D ultrasound image is the standard trans-thalamic plane (Feature A), then
segments the fetal head, measures head circumference (HC) and estimates how reliable that measurement is by
test-time augmentation (Feature B). A second modality, a CTG record, is classified by an XGBoost model with SHAP
explanations, and all outputs are combined into a rule-based screening summary (Feature C).
This document covers the system design, key decisions and the diagrams (architecture, flowchart, UML use-case,
class and sequence diagrams, DFD levels 0 and 1). Diagrams are Mermaid and render on GitHub.

> Academic research prototype, not a medical device.

## 1. System architecture

```mermaid
flowchart LR
    U([User / researcher]) -->|browser| FE[Next.js frontend<br/>frontend/]
    U -->|browser| DB[Streamlit dashboard<br/>dashboard/ - fallback]
    FE -->|HTTP multipart / JSON<br/>/api/v1/*| API[FastAPI backend<br/>backend/app]
    API --> SVC[Services<br/>image_service, model_service]
    SVC --> PG[ml.planes.PlaneClassifier<br/>plane gate]
    PG --> PM[(Plane checkpoint<br/>outputs/plane_classifier_best.pth)]
    SVC --> INF[ml.inference.Segmenter]
    SVC --> TTA[ml.uncertainty.tta_predict<br/>8 augmented passes]
    TTA --> INF
    TTA --> CAL[(Validation thresholds<br/>outputs/uncertainty_calibration.json)]
    API --> CTGS[ctg_service]
    CTGS --> CTGM[ml.ctg.CTGModel<br/>XGBoost Booster + SHAP]
    CTGM --> CK2[(CTG model<br/>outputs/ctg_xgb.json)]
    CTGS --> FUS[ml.fusion.summarize<br/>rules R1-R6]
    DB --> INF
    INF --> M[(U-Net checkpoint<br/>outputs/hc18_unet_best.pth)]
    INF --> GEO[ml.geometry<br/>ellipse fit + Ramanujan HC]
    subgraph Offline pipeline - ml/
      D[(HC18 dataset<br/>data/raw/HC18)] --> MS[masks.py<br/>fill outline]
      MS --> SP[split.py<br/>group split]
      SP --> TR[train.py<br/>GPU, AMP]
      TR --> M
      M --> EV[evaluate.py<br/>Dice, IoU, HC mm]
    end
    subgraph Offline plane pipeline - ml/
      FP[(FETAL_PLANES_DB<br/>data/raw/FETAL_PLANES_DB)] --> TP[train_planes.py<br/>patient split, EfficientNet-B0,<br/>temperature + threshold on val]
      TP --> PM
      PM --> EP[evaluate_planes.py<br/>official test, ECE, HC18 acceptance]
    end
    subgraph Offline CTG pipeline - ml/
      CD[(UCI CTG.xls<br/>data/raw/CTG)] --> TC[train_ctg.py<br/>stratified split, LR / RF / XGBoost,<br/>balanced weights from train, SHAP]
      TC --> CK2
    end
```

| Layer | Technology | Owner |
|---|---|---|
| Data pipeline | Python, OpenCV, pandas | Pujari Varshitha |
| Model training and evaluation | PyTorch 2.5 (CUDA 12.1), U-Net, EfficientNet-B0 plane classifier | Nachiketha K G |
| CTG model and tests | scikit-learn, XGBoost, SHAP | Yashwanth Gowda M N |
| Backend API and integration | FastAPI, Pydantic v2, Uvicorn | Vishwas K |
| Frontend | Next.js 15, TypeScript, Tailwind, shadcn/ui | Rakshitha V |
| Requirements, design docs, testing | pytest, Playwright, GitHub Actions | Yashwanth Gowda M N |

## 2. Design decisions

Short records are in [`adr/`](adr/).

1. **One shared inference module (`ml/inference.py`).** The API, the Streamlit dashboard and the evaluation
   script all call the same `Segmenter`, so preprocessing, thresholding and HC geometry cannot drift apart.
   (Before this, the dashboard used a different resize and a pixel-contour perimeter.)
2. **Fill annotations at original resolution, then resize** (ADR-0001). Fixes the empty-mask bug.
3. **Group-level split by filename prefix** (ADR-0002). Repeat scans of the same fetus stay in one split.
4. **Measure HC at original resolution.** The 256 × 256 network input distorts the aspect ratio, so the mask
   is resized back before fitting the ellipse; pixel spacing is isotropic only there. HC = Ramanujan
   perimeter × pixel spacing (ADR-0003).
5. **Load the model once** in the FastAPI lifespan and run inference in a threadpool; the event loop stays free.
6. **Validate uploads defensively**: declared type (415), chunked size limit (413) before decoding,
   `PIL.Image.verify` (400), actual format (415), pixel limits (400). One error JSON shape for every error.
7. **Pixel spacing is an explicit input.** The model cannot know mm/px; HC is reported only when spacing is
   given. For HC18 demo images the frontend fills it from a SHA-256 lookup table.
8. **Clinical context only from a verified standard** (ADR-0005). The optional gestational age is used to place
   the measured HC on the INTERGROWTH-21st fetal standard; equations from the source paper, verified against
   all 189 cells of the official table.
9. **Standard-plane gate before measurement** (ADR-0006). A two-head EfficientNet-B0 trained on
   FETAL_PLANES_DB returns calibrated plane probabilities; if P(trans-thalamic) is below the validation-chosen
   threshold, the measurement is withheld unless the caller overrides. The gate is optional at runtime
   (`unavailable` if its checkpoint is missing).
10. **Uncertainty by test-time augmentation** (ADR-0007). Eight augmented passes of the same U-Net, each
   transform undone before combining, give HC ± SD, an uncertainty map and a "low reliability, re-scan" flag
   whose thresholds were set on the validation split.
11. **Multi-page "Sonography Console" UI** (ADR-0009): landing page plus Scan, CTG, Report and Evidence pages
   sharing one in-memory session.
12. **CTG branch and rule-based fusion** (ADR-0008). XGBoost on the 21 UCI CTG features (loaded as a plain
   Booster truncated to the early-stopping iteration, so predictions and SHAP use the same trees). The
   screening summary is rule-based (a pathological CTG always escalates) because the ultrasound and CTG data
   come from different patients; learned fusion is Phase 2.
13. **Segmentation quality indicators are heuristics** (confidence, uncertain-boundary fraction, border contact,
   ellipse-fit ratio), shown as warnings, not as calibrated uncertainty.

## 3. Flowchart: analysing one image

```mermaid
flowchart TD
    A([Upload image + optional pixel spacing]) --> B{Declared type PNG/JPEG?}
    B -- no --> E415[415 Unsupported media type]
    B -- yes --> C{Size <= 10 MB?<br/>read in 1 MB chunks}
    C -- no --> E413[413 File too large]
    C -- yes --> D{PIL verify OK and<br/>format PNG/JPEG?}
    D -- corrupted --> E400[400 Invalid image]
    D -- other format --> E415
    D -- yes --> F{Model loaded?}
    F -- no --> E503[503 Model not loaded]
    F -- yes --> PC{Plane model loaded?}
    PC -- no --> PU[plane_check = unavailable]
    PC -- yes --> PG[Letterbox 320, EfficientNet-B0,<br/>calibrated P brain x P TT]
    PG --> PT{P TT >= threshold?}
    PT -- yes --> PP[plane_check = pass]
    PT -- no --> PW[plane_check = warn]
    PW --> OV{override_plane_check?}
    OV -- no --> WH[measurement_withheld = true:<br/>HC, area, centile null]
    OV -- yes --> PP2[overridden = true]
    PU --> G
    PP --> G
    PP2 --> G
    WH --> G
    G[Grayscale, resize 256x256, /255]
    G --> H[U-Net -> sigmoid probability]
    H --> I[Threshold 0.5, resize to original size,<br/>keep largest component]
    I --> J{Ellipse fit possible?}
    J -- no --> K[head_detected = false, flag]
    J -- yes --> L[Fit ellipse, Ramanujan perimeter]
    L --> M{Pixel spacing given?}
    M -- yes --> N[HC mm = perimeter x spacing, area mm2]
    M -- no --> O[HC = null, flag]
    N --> W2{Withheld?}
    W2 -- yes --> P
    W2 -- no --> GA{Gestational age given?}
    GA -- yes --> CT[INTERGROWTH-21st z-score and centile]
    GA -- no --> UQ
    CT --> UQ
    UQ{uncertainty = true?} -- yes --> TT[8 TTA passes, undo transforms,<br/>HC SD, CV, disagreement, SD map]
    TT --> RL{CV or disagreement above<br/>validation threshold?}
    RL -- yes --> LR[low_reliability = true,<br/>Low reliability, re-scan flag]
    RL -- no --> P
    LR --> P
    UQ -- no --> P
    P[Quality flags, overlay + mask PNG]
    O --> P
    K --> P
    P --> Q([200 JSON response])
```

## 4. UML: use-case diagram

```mermaid
flowchart LR
    R([Researcher / student])
    T([Team developer])
    subgraph FetalAI
      UC1((Upload ultrasound image))
      UC2((Enter or auto-fill pixel spacing))
      UC0((Check scan plane))
      UC3((View segmentation and HC))
      UC4((Review quality warnings))
      UC9((Override plane warning))
      UC10((View HC +- SD and uncertainty map))
      UC11((Enter or load a CTG record))
      UC12((View CTG class and SHAP factors))
      UC13((View combined screening summary))
      UC5((Print / download report))
      UC6((View model metrics))
      UC7((Train and evaluate model))
      UC8((Call REST API))
    end
    R --- UC1 & UC2 & UC3 & UC4 & UC5 & UC6 & UC9 & UC10 & UC11 & UC12 & UC13
    T --- UC7 & UC8
    UC3 -. includes .-> UC0
    UC3 -. includes .-> UC4
    UC9 -. extends .-> UC0
    UC12 -. includes .-> UC11
    UC13 -. includes .-> UC3
```

## 5. UML: class diagram (backend and ML core)

```mermaid
classDiagram
    class Settings {
      +Path checkpoint_path
      +str device
      +float max_upload_mb
      +list cors_origins
      +max_upload_bytes() int
    }
    class ModelService {
      +Settings settings
      +Segmenter segmenter
      +PlaneClassifier planes
      +str load_error
      +ready() bool
      +load()
      +info() ModelInfo
      +dict uncertainty_calibration
      +plane_check(gray, override) PlaneCheck
      +predict(gray, pixel_size_mm, ga, override, uncertainty) PredictionResponse
    }
    class TTAResult {
      +list hc_passes_mm
      +float hc_mean_mm
      +float hc_sd_mm
      +float hc_cv_percent
      +float disagreement_fraction
      +ndarray std_map
    }
    class Uncertainty {
      +int n_passes
      +float hc_sd_mm
      +bool low_reliability
      +list reasons
    }
    class PlaneClassifier {
      +float t_plane
      +float t_sub
      +float threshold
      +probabilities(gray) tuple
      +check(gray) PlaneResult
    }
    class PlaneNet {
      +features
      +plane_head
      +subplane_head
      +forward(x) tuple
    }
    class PlaneCheck {
      +str status
      +str plane
      +float tt_probability
      +float threshold
      +bool overridden
      +str message
    }
    class Segmenter {
      +device
      +int image_size
      +dict meta
      +probability_map(gray) ndarray
      +predict(gray, pixel_size_mm) Prediction
    }
    class UNet {
      +forward(x) Tensor
    }
    class Prediction {
      +ndarray probability
      +ndarray mask
      +EllipseFit ellipse
      +float hc_mm
      +float mean_confidence
      +float uncertain_fraction
    }
    class EllipseFit {
      +float cx
      +float cy
      +float semi_major
      +float semi_minor
      +float angle_deg
      +float perimeter_px
      +hc_mm(pixel_size_mm) float
    }
    class PredictionResponse {
      +PlaneCheck plane_check
      +bool measurement_withheld
      +Uncertainty uncertainty
      +float head_circumference_mm
      +EllipseParams ellipse
      +Quality quality
      +str mask_png_base64
      +str overlay_png_base64
    }
    class CTGService {
      +CTGModel model
      +load()
      +predict(features) CTGPrediction
      +examples() list
      +summary(request) ScreeningSummaryResponse
    }
    class CTGModel {
      +Booster booster
      +TreeExplainer explainer
      +predict(features, top_k) CTGResult
    }
    class ScreeningSummary {
      +str level
      +list drivers
      +list not_assessed
      +list caveats
    }
    class APIError {
      +int status_code
      +str code
      +str message
    }
    APIError <|-- InvalidImageError
    APIError <|-- FileTooLargeError
    APIError <|-- UnsupportedMediaTypeError
    APIError <|-- ModelNotLoadedError
    ModelService --> Settings
    ModelService --> Segmenter
    ModelService --> PlaneClassifier
    PlaneClassifier --> PlaneNet
    ModelService ..> PlaneCheck : builds
    ModelService ..> TTAResult : tta_predict
    ModelService ..> Uncertainty : builds
    Segmenter --> UNet
    Segmenter ..> Prediction : creates
    Prediction --> EllipseFit
    ModelService ..> PredictionResponse : builds
    CTGService --> CTGModel
    CTGService ..> ScreeningSummary : summarize()
```

## 6. UML: sequence diagram for `POST /api/v1/predict`

```mermaid
sequenceDiagram
    actor User
    participant FE as Next.js frontend
    participant R as routes.predict
    participant IS as image_service
    participant MS as ModelService
    participant PC as PlaneClassifier (ml)
    participant SG as Segmenter (ml)
    participant G as ml.geometry
    User->>FE: drop image (+ pixel spacing auto-filled)
    FE->>R: POST /api/v1/predict (multipart)
    R->>R: require_model() (503 if not loaded)
    R->>IS: check_declared_type (415)
    R->>IS: read_limited 1 MB chunks (413)
    R->>IS: decode_grayscale: verify, format, size (400/415)
    IS-->>R: uint8 grayscale array
    R->>MS: predict(gray, spacing, GA, override) in threadpool
    MS->>PC: check(gray)
    PC-->>MS: calibrated plane / sub-plane probabilities, P(TT), pass?
    MS->>MS: plane_check block, withhold HC if warn and no override
    MS->>SG: predict(gray, spacing)
    SG->>SG: resize 256, U-Net, sigmoid, threshold
    SG->>G: resize to original, largest component, fit ellipse
    G-->>SG: EllipseFit (perimeter px)
    SG-->>MS: Prediction (mask, ellipse, HC mm, confidence)
    opt uncertainty = true (default)
      MS->>SG: 8 augmented passes (one batch on GPU), undo each transform
      SG-->>MS: per-pass HC, SD map, disagreement
      MS->>MS: compare with validation thresholds, low-reliability flag
    end
    MS->>MS: quality flags, overlay PNG, mask PNG, uncertainty PNG (base64)
    MS-->>R: PredictionResponse
    R-->>FE: 200 JSON
    FE-->>User: plane-check badge first, then images, metric cards, warnings, report
    opt plane check warned
      User->>FE: Measure anyway (override)
      FE->>R: POST /api/v1/predict (override_plane_check=true)
    end
```

### Sequence: CTG record and screening summary

```mermaid
sequenceDiagram
    actor User
    participant FE as Next.js frontend
    participant CR as ctg_routes
    participant CS as CTGService
    participant CM as CTGModel (XGBoost + SHAP)
    participant FU as ml.fusion
    User->>FE: Load example (held-out test record) or type 21 features
    FE->>CR: GET /api/v1/ctg/examples
    FE->>CR: POST /api/v1/ctg/predict (JSON)
    CR->>CR: Pydantic validation (422 on bad input), require_ctg (503)
    CR->>CS: predict(features)
    CS->>CM: booster.predict, TreeExplainer.shap_values
    CM-->>CS: class probabilities, top-5 SHAP factors
    CS-->>FE: CTGPrediction
    FE->>CR: POST /api/v1/screening/summary (plane status, HC z, reliability, CTG class)
    CR->>FU: summarize(rules R1-R6)
    FU-->>CR: level, drivers, not assessed, caveats
    CR-->>FE: ScreeningSummaryResponse
    FE-->>User: CTG card, summary card (also in the printed report)
```

## 7. Data-flow diagram, level 0 (context)

```mermaid
flowchart LR
    U[User] -- "ultrasound image, pixel spacing, GA, override, CTG features" --> P0(("P0 FetalAI system"))
    P0 -- "plane check, mask, overlay, HC mm +- SD, uncertainty map, reliability, CTG class, SHAP, summary, report" --> U
    DS[(HC18 dataset)] -- "images, outlines, reference HC" --> P0
    FPD[(FETAL_PLANES_DB)] -- "images, plane labels, patient IDs" --> P0
    CTGD[(UCI Cardiotocography)] -- "21 features, NSP labels" --> P0
    P0 -- "model metrics" --> U
```

## 8. Data-flow diagram, level 1

```mermaid
flowchart LR
    U[User]
    DS[(D1 HC18 dataset)]
    SPL[(D2 split CSV)]
    CK[(D3 model checkpoint)]
    MET[(D4 metrics JSON / CSV)]
    FPD[(D5 FETAL_PLANES_DB)]
    PCK[(D6 plane checkpoint)]

    P1(("P1 Prepare data<br/>fill masks, split"))
    P2(("P2 Train model"))
    P3(("P3 Evaluate model"))
    P4(("P4 Validate upload"))
    P5(("P5 Segment and measure"))
    P6(("P6 Present results"))
    P7(("P7 Train, calibrate and<br/>evaluate plane gate"))
    P8(("P8 Check plane"))
    P9(("P9 TTA uncertainty"))
    UCAL[(D7 uncertainty thresholds)]
    CTGD[(D8 UCI CTG dataset)]
    CTGK[(D9 CTG model)]
    P10(("P10 Train and evaluate<br/>CTG models"))
    P11(("P11 Classify CTG + SHAP"))
    P12(("P12 Screening summary<br/>rules R1-R6"))

    DS -- images, outlines --> P1
    P1 -- group split --> SPL
    SPL -- train/val lists --> P2
    DS -- images, masks --> P2
    P2 -- best weights --> CK
    CK --> P3
    SPL -- test list --> P3
    P3 -- Dice, IoU, HC error --> MET
    U -- image file, spacing --> P4
    FPD -- images, labels, patient split --> P7
    P7 -- weights, temperatures, threshold --> PCK
    P7 -- plane metrics --> MET
    P4 -- grayscale array --> P8
    PCK -- weights --> P8
    P8 -- plane_check, withhold? --> P5
    P4 -- grayscale array --> P5
    CK -- weights --> P5
    P5 -- mask, ellipse, HC, flags --> P6
    P4 -- grayscale array --> P9
    CK -- weights --> P9
    P3 -- validation P90 thresholds --> UCAL
    UCAL --> P9
    P9 -- HC SD, uncertainty map, reliability --> P6
    CTGD -- features, NSP --> P10
    P10 -- XGBoost model --> CTGK
    P10 -- CTG metrics --> MET
    U -- CTG features --> P11
    CTGK -- model --> P11
    P11 -- class, probabilities, SHAP --> P6
    P6 -- plane status, HC z, reliability, CTG class --> P12
    P12 -- level, drivers, caveats --> P6
    MET -- test metrics --> P6
    P6 -- overlay, metrics, report --> U
```

## 9. Repository layout

```text
ml/         masks, metrics, dataset, split, model, train, evaluate, geometry, inference, intergrowth,
            planes, train_planes, evaluate_planes (Feature A), uncertainty, evaluate_uncertainty (Feature B),
            ctg, train_ctg, fusion (Feature C)
backend/    FastAPI app: api/ (routes, deps), core/ (config, errors, logging), schemas/, services/
frontend/   Next.js app: src/app (/, /scan, /ctg, /report, /evidence), src/components (landing/, scan/, ctg/,
            report/, evidence/, ui/), src/lib (API client, shared session, hooks, pixel-size lookup)
dashboard/  Streamlit research dashboard (uses ml.inference)
scripts/    dataset check, mask figures, pixel-size export, UI end-to-end check, log renderer, launchers
tests/      pytest: ML unit tests and backend/ API tests
docs/       this file, API, dataset card, model card, testing, roadmap, ADRs, screenshots, evidence
```
