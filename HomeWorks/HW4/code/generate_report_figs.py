import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from fairness import load_data, train_baseline_model, accuracy, disparate_impact, zemel_proxy_fairness
from neural_cleanse import load_model, reconstruct_trigger, detect_outlier_scales

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# --- Q3: Fairness metrics and comparison plot ---
Xdf, y = load_data(os.path.join(os.path.dirname(__file__), 'data.csv'))
sensitive = Xdf['gender'].to_numpy()
Xnum = Xdf.select_dtypes(include=[np.number]).drop(columns=['gender']).to_numpy()
Xtrain, Xtest, ytrain, ytest, sens_train, sens_test = train_test_split(Xnum, y.to_numpy(), sensitive, test_size=0.3, random_state=0)
clf = train_baseline_model(Xtrain, ytrain)
Xs_test = clf._scaler.transform(Xtest)
yproba = clf.predict_proba(Xs_test)[:,1]
ypred = (yproba >= 0.5).astype(int)
# ytest is a numpy array from train_test_split -> do not call .to_numpy()
acc_base = accuracy(ytest, ypred)
di_base = disparate_impact(ytest, ypred, sens_test)
zem_base = zemel_proxy_fairness(Xs_test, ypred, sens_test)

# Apply simple promotion/demotion on test set (use same k heuristic)
from fairness import apply_promotion_demotion, retrain_with_swapped_labels
swap_mask = apply_promotion_demotion(Xs_test, yproba, ytest, sens_test, k=10)
# retrain on swapped labels using the training set as demo (we'll simulate by flipping some training labels)
# create swap mask for train (small fraction)
swap_mask_train = np.zeros(Xtrain.shape[0], dtype=bool)
swap_mask_train[: min(10, swap_mask_train.size)] = True
clf_fair = retrain_with_swapped_labels(Xtrain, ytrain, swap_mask_train)
Xs_test2 = clf_fair._scaler.transform(Xtest)
yproba2 = clf_fair.predict_proba(Xs_test2)[:,1]
ypred2 = (yproba2 >= 0.5).astype(int)
acc_fair = accuracy(ytest, ypred2)
di_fair = disparate_impact(ytest, ypred2, sens_test)
zem_fair = zemel_proxy_fairness(Xs_test2, ypred2, sens_test)

# Plot comparison bar chart
labels = ['Accuracy', 'Disparate Impact', 'Zemel-proxy']
base_vals = [acc_base, di_base, zem_base]
fair_vals = [acc_fair, di_fair, zem_fair]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(6,3.5))
ax.bar(x - width/2, base_vals, width, label='Base')
ax.bar(x + width/2, fair_vals, width, label='Fair')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(1.0, max(base_vals + fair_vals) * 1.2))
ax.legend()
for i, (b, f) in enumerate(zip(base_vals, fair_vals)):
    ax.text(i - width/2, b + 0.02, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fairness_comparison.png'), dpi=150)
print('Saved fairness_comparison.png')

# --- Q1: Neural Cleanse demo and mask visualization ---
model = load_model(demo=True, device='cpu')
# prepare tiny random dataset (demo)
import torch
from torch.utils.data import TensorDataset, DataLoader
X = torch.rand(200,1,28,28)
y = torch.randint(0,10,(200,))
ds = TensorDataset(X,y)
loader = DataLoader(ds, batch_size=32)
scales = []
masks = []
for lbl in range(10):
    m, p, s = reconstruct_trigger(model, loader, lbl, device='cpu', steps=200, lr=0.2)
    scales.append(s)
    masks.append(m.squeeze().numpy())
attacked = detect_outlier_scales(scales)
print('Detected attacked (demo):', attacked)
# save mask image for attacked label
mask_img = masks[attacked]
plt.figure(figsize=(3,3))
plt.imshow(mask_img, cmap='gray')
plt.axis('off')
plt.title(f'reconstructed mask (label={attacked})')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'trigger_reconstructed.png'), dpi=150)
print('Saved trigger_reconstructed.png')

# --- Q2: Privacy example numbers (print to stdout for manual copy into report) ---
from privacy import laplace_scale, laplace_cdf_threshold
print('\nPrivacy example numbers (place into report):')
print('Laplace scale (example avg sensitivity 400):', laplace_scale(400, 1.0))
print('P(noisy > 505) with sensitivity=1, eps=0.5:', laplace_cdf_threshold(505, 500, 1.0, 0.5))

# print final metrics for insertion into LaTeX
print('\nMETRICS_SUMMARY:')
print(f'ACC_BASE={acc_base:.4f} DI_BASE={di_base:.4f} ZEM_BASE={zem_base:.4f}')
print(f'ACC_FAIR={acc_fair:.4f} DI_FAIR={di_fair:.4f} ZEM_FAIR={zem_fair:.4f}')
