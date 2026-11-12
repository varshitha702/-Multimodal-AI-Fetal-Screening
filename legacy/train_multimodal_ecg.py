import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from zipfile import ZipFile
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from config import CLASS_NAMES, IMAGE_SIZE, ECG_LENGTH, METRICS_PATH, MODEL_DIR, WEIGHTS_PATH, CLASS_NAMES_PATH
from model import MultimodalFetalModel

def read_image(path_text):
    if "::" in path_text:
        archive_name, member_name = path_text.split("::", 1)
        with ZipFile(archive_name) as zf, zf.open(member_name) as stream:
            return Image.open(stream).convert("RGB").copy()
    return Image.open(path_text).convert("RGB")

def read_ecg(path_text):
    df = pd.read_csv(path_text)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError(f"No numeric ECG column found: {path_text}")
    x = numeric.iloc[:, 0].dropna().to_numpy(dtype=np.float32)
    if len(x) < 64 or not np.isfinite(x).all():
        raise ValueError(f"Invalid ECG signal in {path_text}")
    old = np.linspace(0, 1, len(x), endpoint=False)
    new = np.linspace(0, 1, ECG_LENGTH, endpoint=False)
    x = np.interp(new, old, x).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.from_numpy(x).view(1, -1)

class PairedDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.labels = {c:i for i,c in enumerate(CLASS_NAMES)}
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if augment:
            ops += [transforms.RandomRotation(5), transforms.ColorJitter(brightness=.08, contrast=.08)]
        ops += [transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform = transforms.Compose(ops)
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = self.transform(read_image(str(row.image_path)))
        ecg = read_ecg(str(row.ecg_path))
        label = self.labels[str(row.label)]
        return image, ecg, label

def make_splits(df, seed):
    if "split" in df.columns and {"train","val","test"}.issubset(set(df.split.astype(str).str.lower())):
        df = df.copy(); df["split"] = df.split.astype(str).str.lower()
        if "patient_id" in df.columns:
            counts = df.groupby("patient_id").split.nunique()
            if (counts > 1).any():
                raise ValueError("Patient leakage: patient_id occurs across multiple splits.")
        return df[df.split=="train"], df[df.split=="val"], df[df.split=="test"], "provided"
    if "patient_id" not in df.columns or df.patient_id.isna().any():
        raise ValueError("Manifest must contain official train/val/test split values or a patient_id for group splitting.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed)
    train_idx, hold_idx = next(splitter.split(df, df.label, groups=df.patient_id))
    train, hold = df.iloc[train_idx], df.iloc[hold_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed)
    val_idx, test_idx = next(splitter2.split(hold, hold.label, groups=hold.patient_id))
    return train, hold.iloc[val_idx], hold.iloc[test_idx], "patient_group"

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for image, ecg, y in loader:
            logits = model(image.to(device), ecg.to(device))
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    return ys, ps

def main():
    p=argparse.ArgumentParser(description="Train paired ultrasound + ECG research model")
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true", help="Downloads ImageNet weights; requires internet.")
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df=pd.read_csv(args.manifest)
    required={"image_path","ecg_path","label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    df=df[df.label.isin(CLASS_NAMES)].copy()
    if df.empty: raise ValueError("No rows match the configured ultrasound-view classes.")
    for col in ["image_path","ecg_path"]:
        missing = [x for x in df[col].astype(str) if not Path(x.split("::")[0]).exists()]
        if missing: raise FileNotFoundError(f"{len(missing)} missing paths in {col}; first: {missing[0]}")
    train_df,val_df,test_df,split_mode=make_splits(df,args.seed)
    if min(len(train_df),len(val_df),len(test_df)) == 0:
        raise ValueError("Train/validation/test split is empty.")
    train_dl=DataLoader(PairedDataset(train_df,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val_dl=DataLoader(PairedDataset(val_df),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    test_dl=DataLoader(PairedDataset(test_df),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MultimodalFetalModel(len(CLASS_NAMES),pretrained=args.pretrained).to(device)
    counts=train_df.label.value_counts()
    weights=torch.tensor([len(train_df)/(len(CLASS_NAMES)*max(1,counts.get(c,0))) for c in CLASS_NAMES],dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    best=-1.0
    MODEL_DIR.mkdir(parents=True,exist_ok=True); METRICS_PATH.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); loss_sum=0.0; seen=0
        for image,ecg,y in train_dl:
            image,ecg,y=image.to(device),ecg.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
                logits=model(image,ecg); loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.item())*len(y); seen += len(y)
        vy,vp=evaluate(model,val_dl,device)
        score=f1_score(vy,vp,average="macro",zero_division=0)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={loss_sum/max(seen,1):.4f}, val_macro_f1={score:.4f}")
        if score>best:
            best=score; torch.save(model.state_dict(),WEIGHTS_PATH)
            CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES,indent=2),encoding="utf-8")
    model.load_state_dict(torch.load(WEIGHTS_PATH,map_location=device,weights_only=True))
    ty,tp=evaluate(model,test_dl,device)
    report=classification_report(ty,tp,labels=list(range(len(CLASS_NAMES))),target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    metrics={"test_accuracy":float(accuracy_score(ty,tp)),
             "test_macro_f1":float(f1_score(ty,tp,average="macro",zero_division=0)),
             "classification_report":report,
             "confusion_matrix":confusion_matrix(ty,tp,labels=list(range(len(CLASS_NAMES)))).tolist(),
             "device":str(device),"best_validation_macro_f1":float(best),
             "class_names":CLASS_NAMES,"split_strategy":split_mode,
             "n_train":len(train_df),"n_validation":len(val_df),"n_test":len(test_df),
             "clinical_validity":"Not established; dataset provenance and endpoint must be independently verified."}
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k!="classification_report"},indent=2))

if __name__=="__main__": main()
