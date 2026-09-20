import tempfile
from fpdf import FPDF
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# 1. تحميل خط المعالجة المحدث v2
pipe = joblib.load("eeg_pipeline_v2.joblib")
model, scaler = pipe["model"], pipe["scaler"]


def generate_pdf_report(diag_text, seizure_pct, num_epochs, psd_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(200, 10, txt="Clinical EEG Diagnostic Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Diagnosis: {diag_text}", ln=True)
    pdf.cell(
        200, 10, txt=f"Seizure Epochs Percentage: {seizure_pct:.1f}%", ln=True
    )
    pdf.cell(200, 10, txt=f"Total Processed Epochs: {num_epochs}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", size=14, style="B")
    pdf.cell(200, 10, txt="Spectral Power Band Distribution (PSD):", ln=True)
    pdf.set_font("Helvetica", size=11)
    for band, val in psd_summary.items():
        pdf.cell(200, 8, txt=f" - {band} Band Power: {val:.4f}", ln=True)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# 2. واجهة التطبيق عبر Streamlit
st.title("Clinical Explainable EEG Seizure Dashboard")
st.write(
    "نظام تشخيص إكلينيكي مدعوم بالذكاء الاصطناعي (GroupKFold Validated + PSD Bands Analysis)"
)

uploaded_file = st.file_uploader("رفع ملف EEG (EDF Format)", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # معالجة ملف EDF
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    raw.filter(0.5, 40.0, verbose=False)
    raw.resample(250, verbose=False)
    raw.rename_channels({ch: ch.replace(".", "").strip() for ch in raw.ch_names})

    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # 1. استخراج الخصائص الزمنية (Mean & Std) لكل القطاعات
    mean_feat = data.mean(axis=-1)
    std_feat = data.std(axis=-1)

    # 2. استخراج طاقة نطاقات التردد (PSD)
    psd = epochs.compute_psd(fmin=0.5, fmax=40.0, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)

    delta = psd_data[:, :, (freqs >= 0.5) & (freqs < 4)].mean(
        axis=-1, keepdims=True
    )
    theta = psd_data[:, :, (freqs >= 4) & (freqs < 8)].mean(
        axis=-1, keepdims=True
    )
    alpha = psd_data[:, :, (freqs >= 8) & (freqs < 13)].mean(
        axis=-1, keepdims=True
    )
    beta = psd_data[:, :, (freqs >= 13) & (freqs <= 30)].mean(
        axis=-1, keepdims=True
    )

    # تجميع الخصائص
    feats = np.hstack(
        [
            mean_feat,
            std_feat,
            delta.squeeze(-1),
            theta.squeeze(-1),
            alpha.squeeze(-1),
            beta.squeeze(-1),
        ]
    )

    # التحقق وتعديل شكل الخصائص لتطابق التوقع
    feats_scaled = scaler.transform(feats)

    # التنبؤ
    preds = model.predict(feats_scaled)
    probs_all = model.predict_proba(feats_scaled)

    # معالجة الاحتمالات بأمان لمنع IndexError
    if probs_all.shape[1] > 1:
        probs = probs_all[:, 1]
    else:
        probs = probs_all[:, 0]

    seizure_pct = np.mean(preds) * 100
    is_seizure = np.mean(preds) > 0.5
    diag = (
        "⚠️ نشاط صرعي محتمل (Seizure Detected)"
        if is_seizure
        else "✅ رسم مخ طبيعي (Normal EEG)"
    )

    psd_summary = {
        "Delta (0.5-4 Hz)": float(delta.mean()),
        "Theta (4-8 Hz)": float(theta.mean()),
        "Alpha (8-13 Hz)": float(alpha.mean()),
        "Beta (13-30 Hz)": float(beta.mean()),
    }

    st.subheader(f"التشخيص النهائي: {diag}")
    st.write(f"نسبة القطاعات المصابة: {seizure_pct:.1f}%")

    # التتبع الزمني للنوبة
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        np.arange(len(probs)) * 2.0,
        probs,
        color="red" if is_seizure else "blue",
        linewidth=2,
    )
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_title("Temporal Seizure Probability Profile")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # تنزيل التقرير الطبي
    pdf_path = generate_pdf_report(diag, seizure_pct, len(preds), psd_summary)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 تنزيل التقرير الطبي المطور (PDF)",
            f,
            file_name="Clinical_EEG_Report.pdf",
        )
