from itertools import combinations

import pandas as pd

from common import CSV_DIR, N_SAMPLES, SAMPLE_FRAC, SAMPLES_FILE, load_B

X, y = load_B()
n_total = len(X)
n_sample = round(SAMPLE_FRAC * n_total)

rows = []
index_sets = []
for i in range(N_SAMPLES):
    idx = X.sample(n=n_sample, replace=False, random_state=i).index.sort_values()
    index_sets.append(set(idx))
    rows.append(pd.DataFrame({"imtis": i, "indeksas": idx}))

samples = pd.concat(rows, ignore_index=True)
samples.to_csv(SAMPLES_FILE, index=False)

print(f"Objektu skaicius B aibeje: {n_total}")
print(f"Imciu sk.: {N_SAMPLES}, objektu imtyje: {n_sample} ({SAMPLE_FRAC:.0%})")
print(f"Failas issaugotas: {SAMPLES_FILE}")

overlaps = [len(a & b) for a, b in combinations(index_sets, 2)]
print(f"Imciu poros: {len(overlaps)}")
print(f"Bendras objektu skaicius imtyse: vidurkis {sum(overlaps) / len(overlaps):.1f}, min {min(overlaps)}, max {max(overlaps)}")

counts = pd.DataFrame({"visa_aibe": y.value_counts()})
for i in range(N_SAMPLES):
    idx = samples.loc[samples["imtis"] == i, "indeksas"]
    counts[f"imtis_{i}"] = y.loc[idx].value_counts()
counts = counts.fillna(0).astype(int)
counts.index.name = "klase"
counts.to_csv(CSV_DIR / "7_imciu_klases.csv")

shares = counts.div(counts.sum(axis=0), axis=1) * 100
deviation = shares.drop(columns="visa_aibe").sub(shares["visa_aibe"], axis=0).abs()
summary = pd.DataFrame({
    "visa_aibe_proc": shares["visa_aibe"].round(2),
    "imciu_min": counts.drop(columns="visa_aibe").min(axis=1),
    "imciu_max": counts.drop(columns="visa_aibe").max(axis=1),
    "max_nuokrypis_proc_p": deviation.max(axis=1).round(2),
})
print()
print("Klasiu pasiskirstymas imtyse:")
print(counts.to_string())
print()
print(f"Failas isaugotas: {CSV_DIR / '7_imciu_klases.csv'}")