#!/usr/bin/env python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb


RANDOM_STATE = 42
TRAIN, TEST  = "train_split_80.csv", "test_split_20.csv"
LABEL        = "three_year_status"
KFOLD        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

FIXED = ["tobacco_smoking_history_indicator", "alcohol_history_documented",
         "WDR5","RELN","LRP1B","NOTCH1","PTPRD","PIK3CA",
         "TP53","FBXW7","CDKN2A","NFE2L2","FAT1"]
PCS   = [f"PC{i}" for i in range(1,41)]
FEATS = FIXED + PCS
# --------------------------------------


tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
X_tr, y_tr = tr[FEATS], tr[LABEL]
X_te, y_te = te[FEATS], te[LABEL]

cat = X_tr.select_dtypes(include=["object","category","bool"]).columns
num = X_tr.select_dtypes(exclude=["object","category","bool"]).columns
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ]), cat),
])


def metrics(prob, y, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0,1]).ravel() if len(np.unique(y))==2 else (0,0,0,0)
    return dict(
        AUROC      = roc_auc_score(y, prob),
        Accuracy   = accuracy_score(y, pred),
        Precision  = precision_score(y, pred, zero_division=0),
        Recall     = recall_score(y, pred, zero_division=0),
        Specificity= tn/(tn+fp) if (tn+fp) else 0.0,
        F1         = f1_score(y, pred, zero_division=0)
    )

def youden_threshold(prob, y):
    t = np.unique(np.sort(prob))
    best, thr = -1, 0.5
    for cut in t:
        pred = (prob>=cut).astype(int)
        tn,fp,fn,tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn) else 0
        spec = tn/(tn+fp) if (tn+fp) else 0
        j = sens + spec - 1
        if j > best:
            best, thr = j, cut
    return thr

def orig_name(enc_name:str):
    """去掉 ColumnTransformer 前缀 & OneHot 后缀"""
    if "__" in enc_name:
        tail = enc_name.split("__",1)[1]          
        return tail.split("=")[0]                 
    return enc_name

def print_top_features(model, feat_names_enc, title, k=20):
    if isinstance(model, RandomForestClassifier):
        imp = pd.Series(model.feature_importances_, index=feat_names_enc)
    else:
        fmap = {f"f{idx}":n for idx,n in enumerate(feat_names_enc)}
        gain = model.get_score(importance_type="gain")
        imp  = pd.Series({fmap[k]: gain.get(k,0.) for k in fmap})
    imp_orig = imp.groupby(lambda s: orig_name(s)).sum().sort_values(ascending=False)
    print(f"\n--- {title}: Top {k} features ---")
    for i,(f,v) in enumerate(imp_orig.head(k).items(),1):
        print(f"{i:>2}. {f:<30s} {v:.4f}")

# ================= 1⃣  Random Forest =================
class_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
rf_pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=800, max_depth=None, min_samples_leaf=2,
        class_weight={0:class_w[0],1:class_w[1]},
        n_jobs=-1, random_state=RANDOM_STATE))
])
rf_pipe.fit(X_tr, y_tr)
rf_prob = rf_pipe.predict_proba(X_te)[:,1]
rf_thr  = 0.4                              
#rf_thr  = youden_threshold(rf_prob,  y_te)
rf_test = metrics(rf_prob, y_te, rf_thr)
rf_cv   = cross_val_score(rf_pipe, X_tr, y_tr, cv=KFOLD,
                          scoring="f1", n_jobs=-1).mean()

print_top_features(rf_pipe.named_steps["clf"],
                   rf_pipe.named_steps["pre"].get_feature_names_out(),
                   "RandomForest")

# ================= 2⃣  XGBoost =================
X_tr_enc = pre.fit_transform(X_tr, y_tr)
X_te_enc = pre.transform(X_te)
neg_pos  = (y_tr==0).sum() / max((y_tr==1).sum(),1)

dtrain, dvalid = xgb.DMatrix(X_tr_enc,label=y_tr), xgb.DMatrix(X_te_enc,label=y_te)
params = dict(
    objective="binary:logistic", eval_metric="logloss", eta=0.05,
    max_depth=4, min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
    gamma=0.5, lambda_=10, alpha=0, seed=RANDOM_STATE, nthread=-1,
    scale_pos_weight=neg_pos
)
evals_res={}
bst = xgb.train(params, dtrain, 2000,
                evals=[(dtrain,"train"),(dvalid,"valid")],
                early_stopping_rounds=30, evals_result=evals_res, verbose_eval=False)
best_round = bst.best_iteration
xgb_prob = bst.predict(dvalid)

#xgb_thr  = youden_threshold(xgb_prob, y_te)
xgb_thr = 0.4
xgb_test = metrics(xgb_prob, y_te, xgb_thr)

print_top_features(bst, pre.get_feature_names_out(), "XGBoost")

xgb_cv_clf = xgb.XGBClassifier(
    n_estimators=best_round+1, learning_rate=0.05,
    max_depth=4, min_child_weight=5,
    subsample=0.7, colsample_bytree=0.7,
    gamma=0.5, reg_lambda=10, reg_alpha=0,
    scale_pos_weight=neg_pos,
    objective="binary:logistic", use_label_encoder=False,
    n_jobs=-1, random_state=RANDOM_STATE
)
xgb_cv = cross_val_score(Pipeline([("pre", pre), ("clf", xgb_cv_clf)]),
                         X_tr, y_tr, cv=KFOLD, scoring="f1", n_jobs=-1).mean()


rows=[]
for name, cvf1, thr, tst in [
        ("RandomForest", rf_cv,  rf_thr, rf_test),
        ("XGBoost",      xgb_cv, xgb_thr, xgb_test)]:
    rows.append({"Model":name, "CV_F1":f"{cvf1:.3f}", "Thr":f"{thr:.2f}",
                 **{k:f"{v:.3f}" for k,v in tst.items()}})

print("\n===== CV & Test (chosen threshold) =====")
print(pd.DataFrame(rows).to_string(index=False))


train_loss = evals_res["train"]["logloss"][:best_round+1]
valid_loss = evals_res["valid"]["logloss"][:best_round+1]
plt.figure(); plt.plot(range(1,best_round+2), train_loss, label="Train")
plt.plot(range(1,best_round+2), valid_loss, label="Valid")
plt.xlabel("Boosting round"); plt.ylabel("Logloss")
plt.title("XGBoost logloss (early‑stop)"); plt.legend(); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)

labels = ["RandomForest","XGBoost"]
f1s  = [float(rows[0]["F1"]), float(rows[1]["F1"])]
aucs = [float(rows[0]["AUROC"]), float(rows[1]["AUROC"])]
x=np.arange(len(labels)); w=0.35
plt.figure(); plt.bar(x-w/2, f1s,w,label="F1"); plt.bar(x+w/2,aucs,w,label="AUROC")
plt.xticks(x,labels); plt.ylabel("Score"); plt.title("Model Performance")
plt.legend(); plt.tight_layout(); plt.savefig("model_perf.png", dpi=300)

print("\nPlots & feature lists complete  →", Path().resolve())
