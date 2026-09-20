import tempfile
from fpdf import FPDF
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# 1. تحميل النموذج والـ Scaler المحدث v2
pipe = joblib.load("eeg_pipeline_v2.joblib")
model, scaler = pipe["model"], pipe["scaler"]


# دالة إنشاء التقرير الطبي - تم تنظيفها من العربي والرموز لتجنب Encoding Error
def generate_pdf_report(
    is_seizure_flag, seizure_pct, num_epochs, psd_summary
):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(200, 10, txt="Clinical EEG Diagnostic Report", ln=True, align="C")
    pdf.ln(5)

    # Diagnosis Text in English
    pdf.set_font("Helvetica", size=12, style="B")
    diag_status = (
        "POSITIVE (Seizure Detected)"
        if is_seizure_flag
        else "NEGATIVE (Normal EEG)"
    )
    pdf.cell(200, 10, txt=f"Diagnosis Result: {diag_status}", ln=True)

    # Details
    pdf.set_font("Helvetica", size=11)
    pdf.cell(
        200, 8, txt=f"Seizure Epochs Percentage: {seizure_pct:.1f}%", ln=True
    )
    pdf.cell(200, 8, txt=f"Total Processed Epochs: {num_epochs}", ln=True)
    pdf.ln(5)

    # PSD Table Title
    pdf.set_font("Helvetica", size=12, style="B")
    pdf.cell(
        200, 10, txt="Spectral Power Band Distribution (PSD):", ln=True
    )

    pdf.set_font("Helvetica", size=11)
    for band, val in psd_summary.items():
        pdf.cell(200, 8, txt=f" - {band}: {val:.4f}", ln=True)

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

    # 1. استخراج الخصائص الزمنية
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

    # التحويل والتنبؤ
    feats_scaled = scaler.transform(feats)
    preds = model.predict(feats_scaled)
    probs_all = model.predict_proba(feats_scaled)

    probs = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]

    seizure_pct = np.mean(preds) * 100
    is_seizure = np.mean(preds) > 0.5

    diag_display = (
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

    # العرض على الشاشة (عربي)
    st.subheader(f"التشخيص النهائي: {diag_display}")
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

    # إنشاء التقرير المطور
    pdf_path = generate_pdf_report(
        is_seizure, seizure_pct, len(preds), psd_summary
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 تنزيل التقرير الطبي المطور (PDF)",
            f,
            file_name="Clinical_EEG_Report.pdf",
        )
