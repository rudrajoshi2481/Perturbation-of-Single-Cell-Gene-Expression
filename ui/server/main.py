import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, "/app/cell_line_Database/EndToEnd/src/model/vae")
from vae_model import VAE

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "/app/cell_line_Database/EndToEnd/src/trash/vae_01/best_model.pt"
TRAIN_DATA  = "/app/cell_line_Database/EndToEnd/src/preprocessing/train_data.pt"
INPUT_DIM   = 18211
HIDDEN_DIM  = 128
LATENT_DIM  = 32
NUM_LAYERS  = 2
DEVICE      = "cpu"

DRUG_NAMES = [
    "Clotrimazole","Mometasone Furoate","Idelalisib","Vandetanib","Bosutinib",
    "Ceritinib","Lamivudine","Crizotinib","Cabozantinib","Flutamide","Dasatinib",
    "Selumetinib","Trametinib","ABT-199 (GDC-0199)","Oxybenzone","Vorinostat",
    "Raloxifene","Linagliptin","Lapatinib","Canertinib","Disulfiram","Vardenafil",
    "Palbociclib","Ricolinostat","Proscillaridin A;Proscillaridin-A","IN1451",
    "Ixabepilone","CEP-18770 (Delanzomib)","RG7112","MK-5108","Resminostat",
    "IMD-0354","Alvocidib","LY2090314","Methotrexate","LDN 193189","Tacalcitol",
    "Colchicine","R428","TL_HRAS26","BMS-387032","CGP 60474","TIE2 Kinase Inhibitor",
    "PD-0325901","Isoniazid","GSK-1070916","Masitinib","Saracatinib","CC-401",
    "Decitabine","Ketoconazole","HYDROXYUREA","BAY 61-3606","Navitoclax",
    "Porcn Inhibitor III","GW843682X","Prednisolone","Tamatinib","Tosedostat",
    "GSK256066","MGCD-265","AZD-8330","RN-486","Amiodarone","RVX-208","GO-6976",
    "Scriptaid","HMN-214","SB525334","AVL-292","BMS-777607","AZD4547","Foretinib",
    "Tivozanib","Quizartinib","IKK Inhibitor VII","UNII-BXU45ZH6LI",
    "Chlorpheniramine","Tivantinib","CEP-37440","TPCA-1","AZ628","OSI-930",
    "AZD3514","Vanoxerine","PF-03814735","MLN 2238","Dovitinib","K-02288",
    "Midostaurin","I-BET151","STK219801","PRT-062607","AT 7867","Sunitinib",
    "Penfluridol","BMS-536924","Perhexiline","BI-D1870","FK 866",
    "Mubritinib (TAK 165)","Doxorubicin","Pomalidomide","Colforsin","Phenylbutazone",
    "Protriptyline","Buspirone","Clomipramine","Alogliptin","Nefazodone","ABT737",
    "Dactolisib","Nilotinib","Defactinib","PF-04691502","GLPG0634","Sgc-cbp30",
    "BX 912","SCH-58261","Ruxolitinib","BAY 87-2243","O-Demethylated Adapalene",
    "YK 4-279","Ganetespib (STA-9090)","SLx-2119","Oprozomib (ONX 0912)",
    "Desloratadine","Pitavastatin Calcium","TR-14035","AT13387","CHIR-99021",
    "RG7090","AMD-070 (hydrochloride)","BMS-265246","Tipifarnib","Imatinib",
    "Topotecan","Clemastine",
    "5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine",
    "CGM-097","TGX 221","Azacitidine","Atorvastatin","Riociguat",
]

# Build name -> idx mapping the same way training did (sorted unique names)
DRUG_NAME_TO_IDX = {name: idx for idx, name in enumerate(sorted(DRUG_NAMES))}


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model() -> VAE:
    model = VAE(
        input_dim=INPUT_DIM,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="VAE Perturbation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    control_genes: List[float]   # 18 211-length vector
    drug_name: str
    n_samples: int = 1           # how many stochastic re-samples
    dropout_rate: float = 0.0    # fraction of output genes to zero (consistency check)


class PredictResponse(BaseModel):
    drug_name: str
    drug_idx: int
    n_samples: int
    samples: List[List[float]]   # shape [n_samples, 18211]
    mean: List[float]            # element-wise mean across samples
    std: List[float]             # element-wise std across samples


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/drugs")
def list_drugs():
    return {"drugs": sorted(DRUG_NAMES)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.drug_name not in DRUG_NAME_TO_IDX:
        raise HTTPException(status_code=400, detail=f"Unknown drug: {req.drug_name}")
    if len(req.control_genes) != INPUT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {INPUT_DIM} gene values, got {len(req.control_genes)}",
        )
    if not (1 <= req.n_samples <= 50):
        raise HTTPException(status_code=400, detail="n_samples must be between 1 and 50")

    drug_idx = DRUG_NAME_TO_IDX[req.drug_name]
    x = torch.tensor(req.control_genes, dtype=torch.float32).unsqueeze(0)  # [1, 18211]
    drug_tensor = torch.tensor([drug_idx], dtype=torch.long)

    samples = []
    with torch.no_grad():
        for _ in range(req.n_samples):
            recon, _, _ = model(x, drug_tensor)
            out = recon.squeeze(0).numpy().tolist()

            # apply stochastic dropout for consistency-check resampling
            if req.dropout_rate > 0.0:
                mask = np.random.rand(len(out)) < req.dropout_rate
                out = [0.0 if mask[i] else v for i, v in enumerate(out)]

            samples.append(out)

    arr = np.array(samples)
    mean = arr.mean(axis=0).tolist()
    std  = arr.std(axis=0).tolist()

    return PredictResponse(
        drug_name=req.drug_name,
        drug_idx=drug_idx,
        n_samples=req.n_samples,
        samples=samples,
        mean=mean,
        std=std,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}
