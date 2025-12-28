# evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path

RESULTS = Path("C:/Users/X/Downloads/models/backend/results.csv")
LABELS = Path("C:/Users/X/Downloads/models/Dataset/clips/labels.csv")  # your labels.csv path
OUT_CM = Path("C:/Users/X/Downloads/models/backend/confusion_matrix.csv")

df_r = pd.read_csv(RESULTS)
df_l = pd.read_csv(LABELS)

# labels.csv format: filepath + binary columns for each label; need to create a single label per file: stutter vs non-stutter
# We treat any clip that has any dysfluency label >0 as Stutter, else Non-Stutter
def is_stutter_row(row):
    # adapt: find your SEP-28k label columns; typical: Block, Prolongation, SoundRep, WordRep, Interjection, NoStutteredWords
    label_cols = [c for c in df_l.columns if c.lower() not in ("filepath","filepath")]
    # ensure numeric
    s = row[label_cols].sum()
    return 0 if row.get("NoStutteredWords",0)==1 else 1

# Build ground truth map: filename -> stutter/nonstutter
df_l['filename'] = df_l['filepath'].apply(lambda p: Path(p).name)
# Create binary truth: 1 = Stutter, 0 = Non-Stutter
def truth_from_row(r):
    # If NoStutteredWords == 1 => fluent
    if 'NoStutteredWords' in r and int(r['NoStutteredWords'])==1:
        return 0
    # if any other label is 1 => stutter
    others = [c for c in r.index if c not in ('filepath','filename') and c!='NoStutteredWords']
    for c in others:
        try:
            if int(r[c])==1:
                return 1
        except Exception:
            pass
    return 0

df_l['truth'] = df_l.apply(truth_from_row, axis=1)

# Merge
df = pd.merge(df_r, df_l[['filename','truth']], left_on='filename', right_on='filename', how='inner')
# map predictions to binary
df['pred_bin'] = df['prediction'].map(lambda x: 1 if str(x).lower().startswith('stutter') else 0)

y_true = df['truth']
y_pred = df['pred_bin']

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("Samples used:", len(df))
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))
# save confusion matrix
import csv
with OUT_CM.open("w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["","Pred=NonStutter","Pred=Stutter"])
    w.writerow(["True=NonStutter", cm[0,0], cm[0,1]])
    w.writerow(["True=Stutter", cm[1,0], cm[1,1]])
print("Saved confusion matrix to", OUT_CM)
