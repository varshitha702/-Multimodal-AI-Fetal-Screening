# Architecture

FetalAI first checks that a 2-D ultrasound image is the standard trans-thalamic plane (Feature A), then
segments the fetal head and measures head circumference (HC).
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
```

| Layer | Technology | Owner |
|---|---|---|
| Data pipeline | Python, OpenCV, pandas | Pujari Varshitha |
| Model training and evaluation | PyTorch 2.5 (CUDA 12.1), U-Net, EfficientNet-B0 plane classifier | Nachiketha K G |
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
10. **Segmentation quality indicators are heuristics** (confidence, uncertain-boundary fraction, border contact,
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
    GA -- no --> P
    CT --> P[Quality flags, overlay + mask PNG]
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
      UC5((Print / download report))
      UC6((View model metrics))
      UC7((Train and evaluate model))
      UC8((Call REST API))
    end
    R --- UC1 & UC2 & UC3 & UC4 & UC5 & UC6 & UC9
    T --- UC7 & UC8
    UC3 -. includes .-> UC0
    UC3 -. includes .-> UC4
    UC9 -. extends .-> UC0
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
      +plane_check(gray, override) PlaneCheck
      +predict(gray, pixel_size_mm, ga, override) PredictionResponse
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
      +float head_circumference_mm
      +EllipseParams ellipse
      +Quality quality
      +str mask_png_base64
      +str overlay_png_base64
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
    Segmenter --> UNet
    Segmenter ..> Prediction : creates
    Prediction --> EllipseFit
    ModelService ..> PredictionResponse : builds
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
    MS->>MS: quality flags, overlay PNG, mask PNG (base64)
    MS-->>R: PredictionResponse
    R-->>FE: 200 JSON
    FE-->>User: plane-check badge first, then images, metric cards, warnings, report
    opt plane check warned
      User->>FE: Measure anyway (override)
      FE->>R: POST /api/v1/predict (override_plane_check=true)
    end
```

## 7. Data-flow diagram, level 0 (context)

```mermaid
flowchart LR
    U[User] -- "ultrasound image, pixel spacing, GA, override" --> P0(("P0 FetalAI system"))
    P0 -- "plane check, mask, overlay, HC mm, quality flags, report" --> U
    DS[(HC18 dataset)] -- "images, outlines, reference HC" --> P0
    FPD[(FETAL_PLANES_DB)] -- "images, plane labels, patient IDs" --> P0
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
    MET -- test metrics --> P6
    P6 -- overlay, metrics, report --> U
```

## 9. Repository layout

```text
ml/         masks, metrics, dataset, split, model, train, evaluate, geometry, inference, intergrowth,
            planes, train_planes, evaluate_planes (Feature A)
backend/    FastAPI app: api/ (routes, deps), core/ (config, errors, logging), schemas/, services/
frontend/   Next.js app: src/app (pages), src/components (+ ui/), src/lib (API client, pixel-size lookup)
dashboard/  Streamlit research dashboard (uses ml.inference)
scripts/    dataset check, mask figures, pixel-size export, UI end-to-end check, log renderer, launchers
tests/      pytest: ML unit tests and backend/ API tests
docs/       this file, API, dataset card, model card, testing, roadmap, ADRs, screenshots, evidence
```
