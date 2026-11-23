"""Build report/main.pdf - CVPR-style two-column report for Assignment 3.

Reads experiment outputs from results/ and figs/ and injects the numbers
directly, so every reported value is consistent with the logs. US Letter,
two columns, Times family, ~9pt body.

Usage:  python report/build_report.py --commit <git-hash-of-freeze-commit>
"""
import csv
import json
import argparse
import re
from xml.sax.saxutils import escape
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "figs"

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Image, Table, TableStyle, NextPageTemplate,
                                FrameBreak, KeepTogether)
from PIL import Image as PILImage

PAGE_W, PAGE_H = letter
COL_W = 3.25 * inch
GUTTER = 0.3125 * inch
ML = MR = (PAGE_W - 2 * COL_W - GUTTER) / 2
MT, MB = 1.0 * inch, 1.125 * inch
COL_H = PAGE_H - MT - MB

INK = colors.HexColor("#111111")
GRAY = colors.HexColor("#444444")


def st(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=10, leading=12,
                textColor=INK, alignment=TA_JUSTIFY, spaceBefore=1.5, spaceAfter=2.5)
    base.update(kw)
    return ParagraphStyle(name, **base)

S = {
    "title": st("title", fontName="Times-Bold", fontSize=18, leading=21,
                alignment=TA_CENTER, textColor=INK, spaceAfter=3),
    "authors": st("authors", fontSize=8.6, leading=10.5, alignment=TA_CENTER, spaceAfter=1),
    "rollinfo": st("rollinfo", fontSize=7.3, leading=9.3, alignment=TA_CENTER,
                   textColor=GRAY, spaceAfter=2),
    "h1": st("h1", fontName="Times-Bold", fontSize=11.3, leading=13.5,
             spaceBefore=5, spaceAfter=2.5, alignment=TA_LEFT, keepWithNext=1),
    "h2": st("h2", fontName="Times-Bold", fontSize=9.6, leading=11.5,
             spaceBefore=3.5, spaceAfter=1.8, alignment=TA_LEFT, keepWithNext=1),
    "body": st("body"),
    "caption": st("caption", fontSize=8.0, leading=9.3, alignment=TA_CENTER,
                  textColor=GRAY, spaceBefore=0.5, spaceAfter=3),
    "ref": st("ref", fontSize=9.0, leading=10.8, alignment=TA_LEFT,
              leftIndent=10, firstLineIndent=-10, spaceAfter=1.5),
}


def body(t):
    return Paragraph(t, S["body"])

def h1(t):
    return Paragraph(t, S["h1"])

def h2(t):
    return Paragraph(t, S["h2"])

def caption(t):
    return Paragraph(t, S["caption"])

def img(path, width=COL_W, max_h=None):
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    if max_h and height > max_h:
        height = max_h
        width = height * w / h
    return Image(str(path), width=width, height=height)

def simple_table(data, col_widths, fontsize=6.1, align_center=True):
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("LEADING", (0, 0), (-1, -1), fontsize + 1.7),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 1.7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (0, 0), (-1, -1), "CENTER" if align_center else "LEFT"),
    ]
    if sum(col_widths) > COL_W:
        col_widths = [width * COL_W / sum(col_widths) for width in col_widths]
    t = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def jload(rel):
    with open(RESULTS / rel) as f:
        return json.load(f)

def read_log(name):
    with open(RESULTS / name / "log.csv") as f:
        return list(csv.DictReader(f))

PCT = lambda x, d=1: f"{100 * x:.{d}f}%"
F4 = lambda x: f"{x:.4f}"
REPO_URL = "https://github.com/devinder-research/ml-assignment3"
SEED = "260004"


def build(commit_hash, repo_url, output):
    q1a = jload("q1/q1a_summary.json")
    q1b = jload("q1/q1b_summary.json")
    mixed = jload("q1/q1b-mixed/metrics.json")
    q2a = jload("q2/q2a_summary.json")
    q2b = jload("q2/q2b_summary.json")
    q2c = jload("q2/q2c_summary.json")
    q3a = jload("q3/q3a_summary.json")
    q3b = jload("q3/q3b_summary.json")
    q3c = jload("q3/q3c_summary.json")
    q4a = jload("q4/q4a_summary.json")
    q4b = jload("q4/q4b_summary.json")
    q4c = jload("q4/q4c_summary.json")

    # figure helper with fallback sizing
    def figure(path, width=COL_W, max_h=None):
        return img(path, width, max_h)

    story = []
    story.append(NextPageTemplate("later"))

    # ================= page 1 =================
    story.append(Paragraph(
        "MLPs, CNNs, Autoencoders, SVMs and<br/>Clustering on CIFAR10", S["title"]))
    story.append(Paragraph("<b>Devinder</b>&nbsp;&nbsp;·&nbsp;&nbsp;Roll number: <b>P26AI0004</b>", S["authors"]))
    story.append(Paragraph(
        f"Repository: <link href='{escape(repo_url)}'>{escape(repo_url)}</link><br/>"
        f"Final repository commit: <b>{commit_hash}</b>",
        S["rollinfo"]))
    story.append(FrameBreak())
    story.append(Paragraph("<b>Abstract</b>", st("abstract_head", fontSize=11,
                                                 alignment=TA_CENTER)))
    story.append(body(
        "<i>We compare classifiers and representations on a fixed CIFAR10 split. A large training–validation "
        "gap explains the baseline MLP's low accuracy. Under mixed labels, validation improves early and "
        "then falls as the MLP fits incorrect targets. A CNN achieves higher clean accuracy and loses much "
        "of the advantage when the pixels are permuted. Denoising improves the fixed CNN's accuracy on "
        "corrupted inputs and frozen probes show that reconstruction error and class information rank "
        "the encoders differently. On the selected autoencoder's features, the SVM recovers more class "
        "structure than label-free clustering.</i>"))

    story.append(h1("1&nbsp;&nbsp;Setup and protocol"))
    story.append(body(
        "<b>Dataset and split.</b> CIFAR10 (10 classes; 50,000 official training images; 10,000 official test "
        "images), obtained as the official tarball (MD5 verified). The official training images were split "
        "<b>once</b> with seed 260004 (all numeric characters of the roll number, in order) using a stratified "
        "<i>sklearn.train_test_split</i> (<i>test_size=5000</i>) into 45,000 train / 5,000 validation; indices "
        "are committed at <font face='Courier' size='6.5'>split/cifar10_split_seed260004.npz</font> (train/val "
        "counts: 4,500/500 per class; overlap 0). The validation set is used <b>only</b> for model, "
        "architecture and hyperparameter selection. The official test set is never used for tuning and is "
        "evaluated once per final configuration. Test images are also used for analysis-only purposes (Q3 "
        "reconstruction grids, Q4 near-boundary inspection), which do not feed back into any choice."))
    story.append(body(
        "<b>Common training budget.</b> No augmentation anywhere. Classifiers: Adam lr 1e-3, 40 epochs, batch "
        "256, cross-entropy, weight decay 0 unless stated (exceptions: the Q1(b) memorisation runs use the "
        "same optimizer, with 400 epochs for the 45k conditions, 300 for the 4.5k random-label probe and 60 for the mixed-label control). "
        "Autoencoders: Adam lr 1e-3, 40 epochs, batch 128, "
        "MSE loss on [0,1] images. Linear probes: Adam lr 1e-3, 30 epochs, batch 1024, on <i>frozen</i> "
        "encoders. SVMs: StandardScaler fit on the 9,000-sample SVM subset only. Seeds are fixed to 260004 for "
        "torch/numpy/random throughout; every run logs per-epoch train/validation metrics to "
        "<font face='Courier' size='6.5'>results/*/log.csv</font> and is reproducible from a clean checkout."))
    story.append(body(
        "<b>Reuse.</b> <b>q2-cnn</b> (Q2a) is reused as the fixed CNN classifier of Q3(b). The "
        "validation-selected clean AE of Q3(a) is reused as the Q3(b) baseline, the frozen probe backbone of "
        "Q3(c), and the fixed representation of Q4. All other models are trained independently."))
    story.append(body(
        "<b>Limitations.</b> Results are from a single split and initialization seed. Hardware and backend "
        "differences can affect numerical reproducibility. The experiments use fixed training budgets and "
        "no data augmentation. Test analysis is descriptive and does not select hyperparameters."))

    # ================= Q1 =================
    story.append(h1("2&nbsp;&nbsp;Q1 — MLP failure modes and memorisation"))
    story.append(h2("2.1&nbsp;&nbsp;Q1(a): Why does an MLP perform poorly on CIFAR10?"))
    story.append(body(
        "<b>Chosen reason (H1):</b> the MLP treats the 3,072 input coordinates as independent, unordered "
        "entities; with no locality or weight-sharing prior it fits by raw interpolation, and it <b>overfits</b> — "
        "training accuracy saturates while validation accuracy plateaus far below it. <b>Competing explanation "
        "(H2):</b> the MLP <b>underfits</b> — training itself stalls (optimization/effective capacity is the "
        "constraint), so training accuracy stays low too."))
    story.append(body(
        "<b>Fair comparison.</b> Identical split, seed, optimizer, learning rate, batch size and 40-epoch "
        "budget; parameter count identical for the two budget-matched models (mlp-base vs mlp-reg), and a third "
        "smaller model isolates the capacity probe. Fairness criterion fixed <i>before</i> seeing test numbers: "
        "same training budget and same order of magnitude parameters, with the H1/H2 contrast isolated by "
        "changing only regularisation."))
    story.append(simple_table(
        [["model", "hidden", "drop / wd", "params", "best val", "test"],
         ["mlp-base", "1024–512", "— / 0", f"{q1a['mlp-base']['params']:,}",
          PCT(q1a['mlp-base']['best_val_acc']), PCT(q1a['mlp-base']['test_acc'])],
         ["mlp-reg", "1024–512", "0.3 / 1e-4", f"{q1a['mlp-reg']['params']:,}",
          PCT(q1a['mlp-reg']['best_val_acc']), PCT(q1a['mlp-reg']['test_acc'])],
         ["mlp-small", "256–128", "— / 0", f"{q1a['mlp-small']['params']:,}",
          PCT(q1a['mlp-small']['best_val_acc']), PCT(q1a['mlp-small']['test_acc'])]],
        col_widths=[0.55*inch, 0.55*inch, 0.60*inch, 0.60*inch, 0.50*inch, 0.44*inch], fontsize=5.8))
    story.append(caption("<b>Table 1.</b> Q1(a) — specs, parameter counts, test accuracy (selected by "
                         "validation only)."))
    story.append(figure(FIGS / "fig_q1a_curves.png", COL_W, 2.05 * inch))
    story.append(caption("<b>Figure 1.</b> Q1(a) — train/validation accuracy. mlp-base overfits: train &gt;&gt; val, "
                         "validation plateaus early; mlp-reg (identical budget) barely lifts validation; mlp-small "
                         "confirms the regime."))
    base_last = float(read_log("q1/mlp-base")[-1]["train_acc"])
    reg_last = float(read_log("q1/mlp-reg")[-1]["train_acc"])
    val_gain = 100 * (q1a["mlp-reg"]["best_val_acc"] - q1a["mlp-base"]["best_val_acc"])
    story.append(body(
        f"<b>Evidence.</b> The baseline ends at {PCT(base_last)} train and "
        f"{PCT(q1a['mlp-base']['best_val_acc'])} peak validation accuracy. That gap supports "
        f"overfitting over underfitting for this baseline. Regularization reduces final train accuracy "
        f"to {PCT(reg_last)} and changes peak validation accuracy by {val_gain:+.1f} percentage points. "
        "The low-capacity diagnostic also develops a large generalization gap. Q1 establishes the "
        "gap. The permutation intervention in Q2 examines the separate hypothesis about spatial structure."))

    story.append(h2("2.2&nbsp;&nbsp;Q1(b): Fitting examples that should not generalise"))
    story.append(body(
        "We train the same high-capacity MLP (3072–1024–512–10; 3.68M parameters; no regularisation) on: "
        "(i) the true labels of the fixed 45k split and (ii) i.i.d. uniform random labels for the same images "
        "(400 epochs, Adam lr 1e-3), plus (iii) i.i.d. random labels for a stratified 4,500-image subset "
        "(300 epochs - a memorisation probe well within model capacity), tracking train and validation "
        "accuracy at every epoch. Validation always uses true labels."))
    story.append(simple_table(
        [["labels", "best val (ep)", "test acc"],
         ["true (45k)", f"{PCT(q1b['true']['best_val_acc'])} (ep {q1b['true']['best_epoch']})", PCT(q1b['true']['test_acc'])],
         ["random (45k)", f"{PCT(q1b['random']['best_val_acc'])} (ep {q1b['random']['best_epoch']})", PCT(q1b['random']['test_acc'])],
         ["random (4.5k)", f"{PCT(q1b['random_4k']['best_val_acc'])} (ep {q1b['random_4k']['best_epoch']})", PCT(q1b['random_4k']['test_acc'])],
         ["mixed (4.5k)", f"{PCT(mixed['best_val_acc'])} (ep {mixed['best_epoch']})", PCT(mixed['test_acc'])]],
        col_widths=[1.05*inch, 1.3*inch, 0.65*inch], fontsize=6.0))
    story.append(caption("<b>Table 2.</b> Q1(b) — memorisation setting, final numbers (curves in Fig. 2)."))
    story.append(figure(FIGS / "fig_q1b_memorization.png", COL_W, 2.15 * inch))
    rand45 = float(read_log("q1/q1b-random")[-1]["train_acc"])
    rand_small = float(read_log("q1/q1b-random_4k")[-1]["train_acc"])
    story.append(caption(
        f"<b>Figure 2.</b> The full random-label run reaches {PCT(rand45)} train accuracy, "
        f"and the 4.5k run reaches {PCT(rand_small)}. Their validation accuracy stays near chance."))
    story.append(body(
        "<b>Mixed-label control.</b> With the same 4,500 training images, replace 2,250 labels "
        "with a uniformly sampled different class. Track how well the model predicts those wrong "
        "targets after each epoch, alongside its accuracy on the separate clean validation set "
        "(Fig. 3). Uniform guesses give 10% target accuracy."))
    story.append(figure(FIGS / "fig_q1b_mixed.png", COL_W, 2.0 * inch))
    story.append(caption("<b>Figure 3.</b> Clean validation accuracy and accuracy on wrong training targets."))
    mixed_log = read_log("q1/q1b-mixed")
    mixed_peak = mixed_log[mixed["best_epoch"] - 1]
    late_target = float(mixed_log[-1]["noisy_target_acc"])
    late_val = float(mixed_log[-1]["val_acc"])
    story.append(body(
        f"<b>Timing.</b> At epoch {mixed['best_epoch']}, clean validation reaches "
        f"{PCT(mixed['best_val_acc'])} while the network predicts "
        f"{PCT(float(mixed_peak['noisy_target_acc']))} of the wrong targets. By epoch {mixed['epochs']}, "
        f"wrong-target accuracy reaches {PCT(late_target)} and validation falls to {PCT(late_val)}. "
        "Useful class structure is present early, while fitting of the wrong labels is still limited. "
        "The trajectories can overlap, so the experiment does not establish an exact onset of memorization [2]."))

    # ================= Q2 =================
    story.append(h1("3&nbsp;&nbsp;Q2 — CNN vs MLP, and the spatial-structure hypothesis"))
    story.append(h2("3.1&nbsp;&nbsp;Q2(a): A fair CNN vs MLP comparison"))
    story.append(body(
        "<b>Fairness criterion (fixed before any test number was seen).</b> Identical split, seed, optimizer "
        "(Adam lr 1e-3), batch (256), epochs (40), no augmentation, no test-time tuning; parameter counts kept "
        "within ±20%. The CNN uses fewer parameters. The architectures may differ in per-update compute. "
        "The CNN uses four 3×3 convolutions (strides 1/2/2/2, padding 1), followed by dense layers "
        "4096–256–10. Both models use ReLUs in hidden layers."))
    story.append(simple_table(
        [["model", "params", "best val", "test acc"],
         ["q2-mlp (3072–512–256–10)", f"{q2a['q2-mlp']['params']:,}", PCT(q2a['q2-mlp']['best_val_acc']), PCT(q2a['q2-mlp']['test_acc'])],
         ["q2-cnn (conv 32-64-128-256)", f"{q2a['q2-cnn']['params']:,}", PCT(q2a['q2-cnn']['best_val_acc']), PCT(q2a['q2-cnn']['test_acc'])]],
        col_widths=[1.35*inch, 0.78*inch, 0.62*inch, 0.55*inch], fontsize=5.9))
    story.append(caption("<b>Table 3.</b> Q2(a) — the fair comparison (the CNN uses 16% fewer parameters)."))
    story.append(body(
        "<b>Result.</b> The CNN exceeds the MLP on validation and on the official test set by a large margin "
        f"({PCT(q2a['q2-cnn']['test_acc'])} vs {PCT(q2a['q2-mlp']['test_acc'])}). <b>Proposed specific reason:</b> "
        "the CNN exploits the <b>spatial inductive bias</b> of images — local connectivity plus weight sharing "
        "(translation equivariance) let it detect the same local parts anywhere, needing fewer effective "
        "parameters and generalising better; the MLP has no such mechanism and spends capacity fitting "
        "position-specific weights."))

    story.append(h2("3.2&nbsp;&nbsp;Q2(b): Spatial structure destroyed, pixel values kept"))
    story.append(body(
        "<b>Transform.</b> A single fixed random permutation of the 3,072 coordinates (seed 260004, committed to "
        "the repo) is applied identically to train, validation and test images; every pixel value is preserved. "
        "<b>Prior prediction (fixed before training):</b> (i) q2-mlp is essentially unaffected, up to re-learning; "
        "(ii) q2-cnn drops to (or below) MLP level, because its locality/weight-sharing prior no longer matches "
        "any spatially coherent structure. Both models were retrained with the Q2(a) protocol; the same "
        "transformation is applied to the corresponding test images."))
    story.append(simple_table(
        [["model", "params", "val (permuted)", "test (permuted)"],
         ["q2-mlp", f"{q2b['q2-mlp-permuted']['params']:,}", PCT(q2b['q2-mlp-permuted']['best_val_acc']),
          PCT(q2b['q2-mlp-permuted']['test_acc'])],
         ["q2-cnn", f"{q2b['q2-cnn-permuted']['params']:,}", PCT(q2b['q2-cnn-permuted']['best_val_acc']),
          PCT(q2b['q2-cnn-permuted']['test_acc'])]],
        col_widths=[0.62*inch, 0.72*inch, 0.78*inch, 0.78*inch], fontsize=6.2))
    story.append(caption("<b>Table 4.</b> Q2(b) — same pixels, spatial structure destroyed. Test used the same "
                         "permutation as training."))
    story.append(figure(FIGS / "fig_q2_clean_vs_permuted.png", 2.9 * inch, 2.25 * inch))
    story.append(caption("<b>Figure 4.</b> Q2(b) — clean vs pixel-permuted test accuracy. The CNN's advantage "
                         "vanishes when spatial structure is removed; the MLP is essentially unchanged."))
    story.append(body(
        "<b>Outcome.</b> The MLP's test accuracy changes from "
        f"{PCT(q2a['q2-mlp']['test_acc'], 2)} to {PCT(q2b['q2-mlp-permuted']['test_acc'], 2)}. "
        f"The CNN drops from {PCT(q2a['q2-cnn']['test_acc'], 2)} "
        f"to {PCT(q2b['q2-cnn-permuted']['test_acc'], 2)}. "
        "The change is consistent with the CNN drawing a larger benefit from local image structure. "
        "Per-channel normalization is applied after permuting the raw values, identically on all splits."))

    story.append(h2("3.3&nbsp;&nbsp;Q2(c): What does the CNN actually rely on? A translation sweep"))
    story.append(body(
        "<b>Two explanations for the CNN's behaviour:</b> <b>E1 — translation invariance:</b> the CNN wins because "
        "its weight sharing makes detectors approximately position-agnostic, so relocating the object does not "
        "matter. <b>E2 — weight sharing / local features:</b> the CNN wins because local parts are reused "
        "efficiently, not because absolute position is irrelevant. <b>Experiment.</b> Test-time shifts with zero padding after normalization (mean-color padding) are separate from the training intervention. We apply "
        "shifts of clean test images by k  in  {0,1,2,4,8} px along each of the four directions, applied to the "
        "Q2(a) models without retraining; accuracy (4-direction mean±std) and softmax-confidence of correct vs "
        "incorrect predictions are recorded for each k, together with per-direction accuracies and mean entropy."))
    story.append(figure(FIGS / "fig_q2c_translation.png", COL_W, 2.05 * inch))
    story.append(caption("<b>Figure 5.</b> Q2(c) — translation sweep. Left: accuracy vs shift; middle: mean "
                         "confidence of correct (solid) and incorrect (dashed) predictions; right: flip-to-error "
                         "rate (fraction of originally-correct test samples made incorrect)."))
    cnn_rows = []
    for k in ["0", "1", "2", "4", "8"]:
        cnn_rows.append([f"k = {k}",
                         PCT(q2c["q2-cnn"][k]["acc"]), PCT(q2c["q2-mlp"][k]["acc"]),
                         f"{q2c['q2-cnn'][k]['flip_to_error']:.3f}",
                         f"{q2c['q2-mlp'][k]['flip_to_error']:.3f}",
                         f"{q2c['q2-cnn'][k]['conf_correct']:.2f} / {q2c['q2-cnn'][k]['conf_wrong']:.2f}"])
    story.append(simple_table(
        [["shift k", "CNN acc", "MLP acc", "CNN flip", "MLP flip", "CNN conf corr/wrong"]] + cnn_rows,
        col_widths=[0.48*inch, 0.52*inch, 0.52*inch, 0.52*inch, 0.52*inch, 0.72*inch], fontsize=5.5))
    story.append(caption("<b>Table 5.</b> Q2(c) — zero-padded translation sweep on the official test set."))
    story.append(body(
        "<b>Error and confidence analysis.</b> At a one-pixel shift, the CNN turns "
        f"{PCT(q2c['q2-cnn']['1']['flip_to_error'])} of its originally correct predictions into errors, "
        f"compared with {PCT(q2c['q2-mlp']['1']['flip_to_error'])} for the MLP. At eight pixels both lose "
        f"accuracy (CNN {PCT(q2c['q2-cnn']['8']['acc'])}, MLP {PCT(q2c['q2-mlp']['8']['acc'])}). "
        f"For the CNN, mean confidence on wrong predictions changes from {q2c['q2-cnn']['0']['conf_wrong']:.2f} "
        f"to {q2c['q2-cnn']['8']['conf_wrong']:.2f}; for the MLP it changes from "
        f"{q2c['q2-mlp']['0']['conf_wrong']:.2f} to {q2c['q2-mlp']['8']['conf_wrong']:.2f}. "
        "These means describe confidence on errors; they do not measure calibration. "
        "The CNN tolerates small shifts better, consistent with partial positional robustness. "
        "Large shifts also crop image content, so this test cannot isolate the effect of position alone. "
        "Together, the permutation and translation results support a contribution from spatial structure."))

    # ================= Q3 =================
    story.append(h1("4&nbsp;&nbsp;Q3 — Autoencoders and learned representations"))
    story.append(h2("4.1&nbsp;&nbsp;Q3(a): Three bottleneck capacities"))
    story.append(body(
        "Identical conv encoder/decoder (conv 32-64-128 s2; fc to/from bottleneck; deconv mirror; sigmoid), "
        "trained on [0,1] images with MSE, Adam lr 1e-3, batch 128, 40 epochs, no augmentation; the only "
        "difference is the flat bottleneck B."))
    rows = []
    for k in ["ae-b32", "ae-b256", "ae-b1024"]:
        v = q3a[k]
        rows.append([str(v["bottleneck"]), f"{v['params']:,}", F4(v["val_mse"]), F4(v["test_mse"])])
    story.append(simple_table(
        [["bottleneck B", "params", "val MSE", "test MSE"]] + rows,
        col_widths=[0.8*inch, 0.95*inch, 0.7*inch, 0.7*inch], fontsize=6.0))
    story.append(caption("<b>Table 6.</b> Q3(a) — three autoencoders, identical except bottleneck. Test MSE "
                         "reported once (selection used validation)."))
    story.append(figure(FIGS / "fig_q3a_valmse.png", 2.5 * inch, 1.9 * inch))
    story.append(caption("<b>Figure 6.</b> Q3(a) — validation reconstruction MSE vs bottleneck size "
                         "(log x-axis)."))
    story.append(figure(FIGS / "fig_q3a_curves.png", COL_W, 1.75 * inch))
    story.append(caption("<b>Figure 7.</b> Q3(a) — validation MSE vs epoch for the three bottleneck sizes "
                         "(identical architecture and budget)."))
    story.append(figure(FIGS / "q3a_recon_grid.png", COL_W, 1.6 * inch))
    story.append(caption("<b>Figure 8.</b> Q3(a) — official test images (top) and reconstructions from the "
                         "validation-selected clean AE (bottom)."))
    sel_key = min(q3a, key=lambda k: q3a[k]["val_mse"])
    story.append(body(
        f"<b>Selection (reconstruction error only).</b> Lowest validation MSE: <b>{sel_key}</b> "
        f"(val {F4(q3a[sel_key]['val_mse'])}, test {F4(q3a[sel_key]['test_mse'])}). This is the obvious pick "
        "under a pure reconstruction criterion and we report it as such; §4.3 revisits whether that is the "
        "<i>right</i> criterion once class information is measured."))

    story.append(h2("4.2&nbsp;&nbsp;Q3(b): Denoising vs clean reconstruction for classification"))
    story.append(body(
        "<b>Corruption.</b> Additive Gaussian noise, sigma = 0.25, on [0,1] images, clipped to [0,1]; applied "
        "with the same distribution during training and testing (fixed test draw, seed 999). The noise is an unstructured "
        "corruption a local, content-preserving reconstruction can plausibly undo. <b>DAE.</b> Same architecture "
        "as the selected clean AE; trained with noisy inputs and <i>clean</i> targets; identical training "
        f"budget; no class labels used. <b>Fixed classifier</b> = q2-cnn (Q2a), trained on clean labelled "
        "data, <b>not</b> fine-tuned here."))
    story.append(simple_table(
        [["condition (10,000 official test images)", "test acc"],
         ["1. clean images (reference)", PCT(q3b["test_acc_clean"])],
         ["2. noisy images, classified directly", PCT(q3b["test_acc_noisy_direct"])],
         [f"3. noisy → clean AE (B={q3b['clean_ae_bottleneck']}) → CNN", PCT(q3b["test_acc_cleanae_recon"])],
         [f"4. noisy → DAE (B={q3b['dae_bottleneck']}) → CNN", PCT(q3b["test_acc_dae_recon"])]],
        col_widths=[2.45*inch, 0.7*inch], fontsize=6.0))
    story.append(caption(f"<b>Table 7.</b> Q3(b) — corruption: Gaussian sigma={q3b['noise_sigma']}. Reconstruction "
                         f"MSE against clean targets on noisy inputs — clean AE: {q3b['clean_ae_test_mse_on_noisy']:.4f}, DAE: "
                         f"{q3b['dae_test_mse_on_noisy']:.4f}."))
    story.append(body(
        "<b>Does denoising recover what the classifier needs?</b> Compare condition 2 with 3/4. The DAE clearly "
        "helps (condition 4) and clearly beats the clean-AE baseline (condition 3): it denoises "
        f"({q3b['dae_test_mse_on_noisy']:.4f} vs {q3b['clean_ae_test_mse_on_noisy']:.4f} MSE against the clean "
        "targets) and restores class evidence the classifier can use — yet it does not fully restore clean-level "
        f"accuracy ({PCT(q3b['test_acc_clean'])} vs {PCT(q3b['test_acc_dae_recon'])}). The DAE closes "
        f"{100 * (q3b['test_acc_dae_recon'] - q3b['test_acc_noisy_direct']) / (q3b['test_acc_clean'] - q3b['test_acc_noisy_direct']):.0f}% "
        "of the corruption gap. <b>Verdict:</b> denoising recovers most, but not yet all, of the information "
        "the classifier needs; it beats both the noisy-direct baseline and the non-denoising autoencoder, "
        "without using class labels."))

    story.append(figure(FIGS / "q3b_recon_grid.png", COL_W, 1.6 * inch))
    story.append(caption("<b>Figure 9.</b> Q3(b) — noisy test images (top) and DAE reconstructions (bottom); "
                         "the corruption is largely undone with class-relevant structure preserved."))
    story.append(h2("4.3&nbsp;&nbsp;Q3(c): Is reconstruction quality = class information?"))
    story.append(body(
        "<b>Protocol.</b> A fresh linear probe is trained for 30 epochs on each frozen encoder's "
        "features from clean training images. The main comparison evaluates both reconstruction and "
        "probe accuracy on the same clean test inputs. A separate noise stress test uses seed 260004."))
    rows = []
    for k, v in sorted(q3c.items(), key=lambda kv: kv[1]["bottleneck"]):
        rows.append([k, str(v["bottleneck"]), F4(v["val_mse"]), F4(v["test_mse"]),
                     PCT(v["probe_val_acc"]), PCT(v["probe_test_acc_clean"])])
    story.append(simple_table(
        [["encoder", "B", "val MSE", "test MSE", "probe val", "probe test"]] + rows,
        col_widths=[0.65*inch, 0.34*inch, 0.51*inch, 0.53*inch, 0.57*inch, 0.62*inch], fontsize=6.3))
    dae = next(v for k, v in q3c.items() if k.startswith("dae"))
    story.append(caption("<b>Table 8.</b> Clean-input reconstruction error and linear-probe accuracy. "
                         f"On the separate noisy test inputs the DAE probe achieves {PCT(dae['probe_test_acc_noisy'])}."))
    story.append(figure(FIGS / "fig_q3c_probe_vs_mse.png", 2.75 * inch, 2.2 * inch))
    story.append(caption("<b>Figure 10.</b> Clean test-set reconstruction error and linear-probe accuracy. "
                         "The preferred direction is up and to the left."))
    best_ae = q3c[sel_key]
    story.append(body(
        "<b>Comparison.</b> Among clean AEs, greater capacity reduces test MSE and improves probe accuracy. "
        f"The DAE achieves {PCT(dae['probe_test_acc_clean'])} probe accuracy at test MSE {F4(dae['test_mse'])}. "
        f"The reconstruction-selected clean AE achieves {PCT(best_ae['probe_test_acc_clean'])} at "
        f"MSE {F4(best_ae['test_mse'])}. The differing rankings show that pixel reconstruction fidelity "
        "and class information available to a linear classifier measure different properties."))

    # ================= Q4 =================
    story.append(h1("5&nbsp;&nbsp;Q4 — Bonus: SVM and clustering on a fixed representation"))
    rep = q4a["representation"]
    story.append(h2("5.1&nbsp;&nbsp;Q4(a): C × kernel sweep for the SVM"))
    story.append(body(
        f"<b>Fixed representation (chosen once, unchanged for all of Q4):</b> the bottleneck activations of "
        f"the validation-selected clean AE (B={rep['bottleneck']}). "
        f"<b>Fixed SVM subset:</b> a stratified {rep['subset_size']:,}-sample subset of the 45k train split "
        f"(seed {rep['subset_seed']}), chosen <i>before</i> any model selection, used for every SVM fit; the "
        "full 45k training set is used for clustering (Q4c). Validation = the fixed 5,000 validation split. "
        "Features standardised (subset statistics only). Kernels: <b>linear</b>, <b>RBF</b> (gamma=scale), "
        "<b>poly</b> degree 3 (gamma=scale); C in {0.01, 0.1, 1, 10, 100} (four orders of magnitude). "
        "Selection rule fixed beforehand: highest validation accuracy; ties -&gt; lower C, then lower SV fraction."))
    sweep_rows = []
    for r in q4a["sweep"]:
        sweep_rows.append([r["kernel"], f"{r['C']:g}", PCT(r["train_acc"]), PCT(r["val_acc"]),
                           f"{r['sv_fraction']:.3f}", f"{r['fit_seconds']:.0f}"])
    story.append(simple_table(
        [["kernel", "C", "train", "val", "SV frac", "fit s"]] + sweep_rows,
        col_widths=[0.55*inch, 0.45*inch, 0.48*inch, 0.48*inch, 0.52*inch, 0.42*inch], fontsize=5.6))
    story.append(caption("<b>Table 9.</b> Q4(a) — complete C × kernel sweep; selection used validation only."))
    story.append(figure(FIGS / "fig_q4_sweep.png", COL_W, 2.05 * inch))
    story.append(caption("<b>Figure 11.</b> Q4(a) — validation accuracy (left) and support-vector fraction "
                         "(right) vs C, per kernel."))
    story.append(figure(FIGS / "fig_q4_confusion.png", 2.5 * inch, 2.4 * inch))
    story.append(caption("<b>Figure 12.</b> Q4(a) — official test confusion matrix of the selected SVM."))
    _best = {}
    for r in q4a["sweep"]:
        _best.setdefault(r["kernel"], r)
        if r["val_acc"] > _best[r["kernel"]]["val_acc"]:
            _best[r["kernel"]] = r
    _per_kernel = "; ".join(f"{k}: best val {PCT(v['val_acc'])} at C={v['C']:g}"
                            for k, v in sorted(_best.items()))
    sel = q4a["selected"]
    fin = q4a["final"]
    story.append(body(
        f"<b>Selection.</b> By the validation rule: <b>kernel={sel['kernel']}, C={sel['C']:g}</b> "
        f"(validation {PCT(sel['val_acc'])}). Per-kernel validation optima (Table 9): {_per_kernel}. "
        f"The official test set was evaluated <b>once</b>: "
        f"<b>test accuracy = {PCT(fin['test_acc'], 2)}</b> (train {PCT(fin['train_acc'])}, val {PCT(fin['val_acc'])}; "
        f"support vectors {fin['n_support_vectors']:,} = {PCT(fin['sv_fraction'])})."))
    linear = [r for r in q4a["sweep"] if r["kernel"] == "linear"]
    rbf = [r for r in q4a["sweep"] if r["kernel"] == "rbf"]
    poly = [r for r in q4a["sweep"] if r["kernel"] == "poly"]
    best_rbf = max(rbf, key=lambda row: row["val_acc"])
    story.append(body(
        f"<b>Regularization.</b> For the linear kernel, training accuracy increases to "
        f"{PCT(linear[-1]['train_acc'])} while validation falls from {PCT(linear[0]['val_acc'])} "
        f"to {PCT(linear[-1]['val_acc'])}. RBF improves through C={best_rbf['C']:g} and then loses "
        f"validation accuracy at C=100 while training approaches 100%. The cubic polynomial kernel "
        f"underfits at low C ({PCT(poly[0]['val_acc'])} validation) and reaches "
        f"{PCT(poly[-1]['val_acc'])} validation at C=100. A high support-vector fraction means many "
        "training examples contribute to the solution. The training–validation gap is the evidence of overfitting."))

    story.append(h2("5.2&nbsp;&nbsp;Q4(b): The three most confused pairs"))
    story.append(body(
        "Rank unordered class pairs by the sum of the two off-diagonal test confusion counts. "
        "For each of the top three pairs, extract the corresponding one-vs-one decision score "
        "from the selected multiclass SVM. Among training support vectors with a nonzero dual "
        "coefficient for that pair and among test examples whose true label is in the pair, "
        "show the six with the smallest absolute score. A positive score votes for the first class. "
        "The predicted labels and test errors in Table 10 and Fig. 13 use the full multiclass prediction."))
    pair_rows = []
    for p in q4b["pairs"]:
        pair_rows.append([" / ".join(p["pair"]), str(p["confusion_mass_test"]),
                          f"{p['near_boundary_test_errors']}/{p['near_boundary_test_n']}",
                          str(p["pair_support_count"]), f"{p['centroid_distance']:.1f}",
                          " / ".join(f"{radius:.1f}" for radius in p["within_class_radius"])])
    story.append(simple_table(
        [["pair", "mutual err.", "near err.", "pair SVs", "centroid d", "RMS radii"]] + pair_rows,
        col_widths=[0.87*inch, 0.49*inch, 0.46*inch, 0.47*inch, 0.48*inch, 0.48*inch], fontsize=6.1))
    story.append(caption("<b>Table 10.</b> Pair confusion, multiclass errors among the six nearest "
                         "test points, and geometry in standardized bottleneck space."))
    for suffix, p in zip("abc", q4b["pairs"]):
        i, j = p["pair_indices"]
        story.append(figure(FIGS / f"q4_pair_{i}_{j}.png", COL_W, 1.65 * inch))
        story.append(caption(f"<b>Figure 13{suffix}.</b> {' vs '.join(p['pair'])}. Top: active training SVs. "
                             "Bottom: near-boundary test images. Titles give the pair score and true → predicted class."))
    sv_text = " ".join(
        f"<b>{'/'.join(p['pair'])}</b> (" + ", ".join(
            f"{row['d_pair']:+.2e}" for row in p["train_support_vectors"]
        ) + ")." for p in q4b["pairs"]
    )
    story.append(body("<b>Training support vectors, signed scores in figure order.</b> " + sv_text))
    nb_rows = []
    for p in q4b["pairs"]:
        for r in p["test_examples"]:
            nb_rows.append([" / ".join(p["pair"]), r["true"], r["pred"], f"{r['d_pair']:+.3e}"])
    story.append(simple_table(
        [["pair", "true", "prediction", "OvO score"]] + nb_rows,
        col_widths=[1.05*inch, 0.66*inch, 0.74*inch, 0.80*inch], fontsize=6.2))
    story.append(caption("<b>Table 11.</b> Official test examples nearest each pairwise boundary."))
    comparisons = ", ".join(
        f"{'/'.join(p['pair'])}: {p['near_boundary_test_errors']}/{p['near_boundary_test_n']} "
        f"near the boundary vs {PCT(p['pair_error_rate_test'])} across pair members" for p in q4b["pairs"]
    )
    story.append(body(
        f"<b>Errors and geometry.</b> {comparisons}. Centroid separations are smaller than the "
        "within-class RMS radii for each shown pair. The feature groups have broad spread relative "
        "to their mean separation. A score near zero only describes the pairwise classifier's vote, "
        "and other pairwise votes can determine the multiclass result."))
    story.append(body(
        "<b>Image examples.</b> The cat misclassified as dog is a tight face crop, while two dogs "
        "misclassified as horse are side-on, full-body views. The automobile misclassified as truck "
        "has a boxy silhouette and strong two-tone contrast. The deer misclassified as bird is a small "
        "brown subject against a vivid green field. These examples illustrate variation in framing "
        "and background near a boundary, without identifying which pixels caused a decision."))

    story.append(h2("5.3&nbsp;&nbsp;Q4(c): Label-free clustering vs the SVM"))
    sil_txt = ", ".join(f"k={k}: {v:+.4f}" for k, v in q4c["val_silhouette"].items())
    meth_scores = q4c.get("method_candidates_val_silhouette", {})
    meth_txt = ", ".join(f"{m}: {v:+.4f}" for m, v in meth_scores.items())
    story.append(body(
        "<b>Representation unchanged.</b> <b>Method and k chosen with training/validation data only</b> — "
        "neither used test labels anywhere in fitting or selection. <b>Method</b> among "
        "{MiniBatchKMeans, full KMeans (n_init=10), diagonal-covariance GMM}, scored by silhouette on a fixed "
        f"1,500-sample validation subsample (seed {SEED}): {meth_txt} → choose "
        f"<b>{q4c['method']}</b>. <b>k</b> in {{10, 20}} on the same subsample: {sil_txt} → "
        f"choose <b>k = {q4c['k_chosen_by_validation']}</b>. "
        "The final fit uses all 45,000 train representations (8 initializations for MiniBatchKMeans); test representations are assigned without "
        "labels; test labels used only afterwards, once, for evaluation."))
    story.append(simple_table(
        [["metric (official test, after assignment)", "value"],
         ["ARI", f"{q4c['test_ARI']:.4f}"],
         ["NMI", f"{q4c['test_NMI']:.4f}"],
         ["AMI", f"{q4c['test_AMI']:.4f}"],
         ["cluster purity", f"{q4c['test_cluster_purity']:.4f}"],
         ["acc. after majority cluster-&gt;class map", PCT(q4c["test_acc_after_majority_mapping"])],
         ["SVM test accuracy (reference)", PCT(q4c["svm_test_acc"])]],
        col_widths=[2.2*inch, 0.8*inch], fontsize=6.0))
    story.append(caption("<b>Table 12.</b> Q4(c) — test-set clustering evaluation (labels used only after "
                         "assignment); SVM reference point in the last row."))
    cont = q4c["contingency_clusters_x_classes"]
    abbrev = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    hdr = ["cluster / majority"] + abbrev
    full = [hdr]
    for i, row in enumerate(cont):
        majority = q4c["cluster_to_class_map"][i] if sum(row) else "empty"
        full.append([f"c{i} / {majority}", *map(str, row)])
    story.append(simple_table(full, col_widths=[0.95*inch] + [0.222*inch]*10, fontsize=4.6))
    story.append(caption("<b>Table 13.</b> Q4(c) — test cluster-&gt;class contingency (rows = clusters named by "
                         "their majority class; columns = classes, abbreviated)."))
    _svm_map, _cl_map = q4c["per_class_recall_svm"], q4c["per_class_recall_clustering"]
    story.append(body(
        f"<b>Quantitative disagreement.</b> For automobile, SVM recall is {PCT(_svm_map['automobile'])}, "
        f"while test-majority-mapped cluster recall is {PCT(_cl_map['automobile'])}. For airplane the "
        f"corresponding numbers are {PCT(_svm_map['airplane'])} and {PCT(_cl_map['airplane'])}. "
        "The contingency table shows how mixed-class clusters and shared majority labels can yield these "
        "differences. Majority mapping uses test labels for descriptive evaluation, and its accuracy "
        "equals purity. ARI and NMI compare assignments with labels without that mapping."))
    _pc_rows = [[c, PCT(_svm_map[c]), PCT(_cl_map[c]), f"{100 * (_svm_map[c] - _cl_map[c]):+.1f}"]
                for c in _cl_map]
    story.append(simple_table(
        [["class", "SVM recall", "KMeans recall", "Δ (pts)"]] + _pc_rows,
        col_widths=[1.15*inch, 0.72*inch, 0.78*inch, 0.62*inch], fontsize=5.8))
    story.append(caption("<b>Table 14.</b> Q4(c) — per-class test recall: supervised SVM vs label-free MiniBatchKMeans "
                         "(majority-mapped clusters). Positive Δ favours the SVM."))

    # ================= reproducibility =================
    story.append(h1("6&nbsp;&nbsp;Reproducibility"))
    story.append(body(
        "<b>Environment.</b> Python 3.10, torch 2.14.1 / torchvision 0.29.2 (Apple M2, MPS); numpy 2.2.6; "
        "scikit-learn 1.7.2; <font face='Courier' size='6.5'>requirements.txt</font> pins everything; "
        "<font face='Courier' size='6.5'>README.md</font> lists the exact command sequence. Split indices and the "
        "pixel permutation are committed (<font face='Courier' size='6.5'>split/</font>). Every run logs "
        "per-epoch metrics under results/ for the classifier and autoencoder training runs. Probe and bonus "
        "results are stored as JSON and the full SVM sweep is stored as CSV."))
    story.append(body(
        "<b>Reused models (identified):</b> <b>q2-cnn</b> — introduced in Q2(a), reused unchanged as the fixed "
        "classifier of Q3(b); <b>validation-selected clean AE</b> (results/q3/ae-*) — introduced in Q3(a), reused "
        "as the Q3(b) baseline, the probe backbone of Q3(c), and the fixed representation of Q4(a–c); the SVM "
        "selected in Q4(a) — reused for the Q4(b) inspection and the Q4(c) reference. No model weights, logits "
        "or test predictions are shared between any other questions; metric plots are regenerated by "
        "<font face='Courier' size='6.5'>scripts/make_figures.py</font>."))
    reuse_rows = [
        ["q2-cnn", "Q2(a)", "Q3(b) fixed classifier; Q2(c) analysis"],
        ["validation-selected clean AE", "Q3(a)", "Q3(b–c); Q4 representation"],
        ["pixel permutation (split file)", "Q2(b)", "Q2(b) test evaluation only"],
        ["selected SVM (Q4a)", "Q4(a)", "Q4(b) hard-pair inspection; Q4(c) reference"],
        ["fixed stratified 9k SVM subset", "Q4(a)", "every SVM comparison, incl. Q4(b)"],
    ]
    story.append(simple_table(
        [["artifact", "introduced", "reused in"]] + reuse_rows,
        col_widths=[1.20*inch, 0.60*inch, 1.50*inch], fontsize=5.7, align_center=False))
    story.append(caption("<b>Table 15.</b> Reused models/components and where they first appeared."))

    story.append(h1("7&nbsp;&nbsp;Conclusion"))
    story.append(body(
        "Learning usable class structure and fitting wrong training targets follow different trajectories. "
        "The spatial interventions explain part of the CNN advantage. Denoising recovers much of the "
        "classification performance lost to Gaussian noise, and probe accuracy distinguishes encoders "
        "whose reconstruction scores would rank them differently. The frozen bottleneck supports "
        "moderate supervised classification and weak recovery of CIFAR10 classes by clustering."))

    # ================= references =================
    story.append(h1("References"))
    refs = [
        "A. Krizhevsky. Learning multiple layers of features from tiny images. <i>Technical report, University of Toronto</i>, 2009. https://www.cs.toronto.edu/~kriz/cifar.html",
        "C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals. Understanding deep learning requires rethinking generalization. <i>ICLR</i>, 2017.",
        "D. P. Kingma, J. Ba. Adam: A method for stochastic optimization. <i>ICLR</i>, 2015.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {r}", S["ref"]))

    # ================= document assembly =================
    doc = BaseDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title="ML Assignment 3 — Devinder (P26AI0004)",
        author="Devinder",
        invariant=1,
    )
    title_height = 1.45 * inch

    def frame(x, y, width, height, name):
        return Frame(x, y, width, height, id=name, leftPadding=0, rightPadding=0,
                     topPadding=0, bottomPadding=0)

    def columns(height):
        return [frame(ML, MB, COL_W, height, "left"),
                frame(ML + COL_W + GUTTER, MB, COL_W, height, "right")]

    title_frame = frame(ML, MB + COL_H - title_height, 2 * COL_W + GUTTER,
                        title_height, "title")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[title_frame, *columns(COL_H - title_height)]),
        PageTemplate(id="later", frames=columns(COL_H)),
    ])
    blocks = []
    i = 0
    while i < len(story):
        item = story[i]
        following = story[i + 1] if i + 1 < len(story) else None
        if isinstance(item, (Image, Table)) and getattr(following, "style", None) is S["caption"]:
            blocks.append(KeepTogether([item, following]))
            i += 2
        else:
            blocks.append(item)
            i += 1
    doc.build(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--output", type=Path, default=REPO / "report" / "main.pdf")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        ap.error("--commit must be a full 40-character Git commit hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.commit, args.repo_url, args.output)


if __name__ == "__main__":
    main()
