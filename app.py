import os
import tempfile
from datetime import datetime
from fpdf import FPDF
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# 1. تحميل النموذج والـ Scaler من إصدار V3 الموحد
@st.cache_resource
def load_v3_pipeline():
    return joblib.load("eeg_pipeline_v3.joblib")

pipeline = load_v3_pipeline()
model, scaler = pipeline["model"], pipeline["scaler"]


# 2. دالة استخراج الـ 18 ميزة التجميعية الموحدة (Channel-Agnostic Engine)
def extract_v3_comprehensive_features(raw):
    raw_clean = raw.copy()
    # High-pass 0.5Hz لإزالة Baseline Drift، وLow-pass 30Hz لمنع تشويش العضلات
    raw_clean.filter(l_freq=0.5, h_freq=30.0, verbose=False)
    raw_clean.resample(sfreq=250, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw_clean, duration=2.0, preload=True, verbose=False
    )
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # أ) الخصائص الزمنية الشاملة عبر الجمجمة (Global Temporal)
    global_mean_time = data.mean(axis=1)  # (n_epochs, n_times)
    mean_feat = global_mean_time.mean(axis=-1, keepdims=True)
    std_feat = global_mean_time.std(axis=-1, keepdims=True)
    max_feat = global_mean_time.max(axis=-1, keepdims=True)  # التقاط الشذوذ البؤري
    min_feat = global_mean_time.min(axis=-1, keepdims=True)

    # ب) الخصائص الترددية الشاملة (Global Spectral Power Density)
    psd = epochs.compute_psd(fmin=0.5, fmax=30.0, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    global_psd = psd_data.mean(axis=1)

    delta = global_psd[:, (freqs >= 0.5) & (freqs < 4)].mean(
        axis=-1, keepdims=True
    )
    theta = global_psd[:, (freqs >= 4) & (freqs < 8)].mean(
        axis=-1, keepdims=True
    )
    alpha = global_psd[:, (freqs >= 8) & (freqs < 13)].mean(
        axis=-1, keepdims=True
    )
    beta = global_psd[:, (freqs >= 13) & (freqs <= 30)].mean(
        axis=-1, keepdims=True
    )

    # ج) التشتت المكاني والانحراف المعياري بين القنوات (Spatial Dispersion for Focal Seizures)
    ch_delta_std = (
        psd_data[:, :, (freqs >= 0.5) & (freqs < 4)]
        .mean(axis=-1)
        .std(axis=-1, keepdims=True)
    )
    ch_theta_std = (
        psd_data[:, :, (freqs >= 4) & (freqs < 8)]
        .mean(axis=-1)
        .std(axis=-1, keepdims=True)
    )
    ch_alpha_std = (
        psd_data[:, :, (freqs >= 8) & (freqs < 13)]
        .mean(axis=-1)
        .std(axis=-1, keepdims=True)
    )
    ch_beta_std = (
        psd_data[:, :, (freqs >= 13) & (freqs <= 30)]
        .mean(axis=-1)
        .std(axis=-1, keepdims=True)
    )

    # د) النسب الترددية التشخيصية والحاكمة للتشويش
    slow_fast_ratio = (delta + theta) / (alpha + beta + 1e-6)
    theta_alpha_ratio = theta / (alpha + 1e-6)
    delta_beta_ratio = delta / (beta + 1e-6)

    # هـ) الفوارق العظمى بين القنوات
    max_ch_diff = psd_data.mean(axis=-1).max(
        axis=-1, keepdims=True
    ) - psd_data.mean(axis=-1).min(axis=-1, keepdims=True)

    X_features = np.hstack([
        mean_feat,
        std_feat,
        max_feat,
        min_feat,
        delta,
        theta,
        alpha,
        beta,
        ch_delta_std,
        ch_theta_std,
        ch_alpha_std,
        ch_beta_std,
        slow_fast_ratio,
        theta_alpha_ratio,
        delta_beta_ratio,
        max_ch_diff,
    ])

    return X_features, psd_data, epochs


# 3. إنشاء التقرير الطبي المعتمد (EHR-Ready PDF Report)
def generate_clinical_pdf_report(
    diag_status,
    seizure_pct,
    num_epochs,
    psd_summary,
    plot_path,
    topomap_path=None,
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(
        200, 10, txt="CLINICAL EEG AUTOMATED DIAGNOSTIC REPORT", ln=True, align="C"
    )
    pdf.set_font("Helvetica", size=9)
    pdf.cell(
        200,
        5,
        txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine Version: V3.0-ChannelAgnostic",
        ln=True,
        align="C",
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", size=12, style="B")
    pdf.cell(200, 8, txt=f"Primary Diagnostic Finding: {diag_status}", ln=True)

    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        200, 6, txt=f"Seizure Burden (Epochs Ratio): {seizure_pct:.1f}%", ln=True
    )
    pdf.cell(
        200, 6, txt=f"Total Analyzed Epochs (2-sec Windows): {num_epochs}", ln=True
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(
        200, 8, txt="Spectral Power Band Distribution (Mean PSD):", ln=True
    )
    pdf.set_font("Helvetica", size=10)
    for band, val in psd_summary.items():
        pdf.cell(200, 5, txt=f" - {band}: {val:.4f} uV^2/Hz", ln=True)
    pdf.ln(5)

    if os.path.exists(plot_path):
        pdf.set_font("Helvetica", size=11, style="B")
        pdf.cell(
            200, 8, txt="Temporal Probability Profile Across Time:", ln=True
        )
        pdf.image(plot_path, x=15, w=180)
        pdf.ln(4)

    if topomap_path and os.path.exists(topomap_path):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(
            200,
            10,
            txt="2D Spatial Localization & Topographic Heatmap:",
            ln=True,
        )
        pdf.image(topomap_path, x=25, w=160)
        pdf.ln(10)

    pdf.ln(10)
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(100, 6, txt="Attending Physician Signature: __________________", ln=False)
    pdf.cell(90, 6, txt="Date: _____________", ln=True)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# 4. واجهة Streamlit الرئيسية
st.set_page_config(
    page_title="Clinical EEG Seizure Dashboard (V3)", layout="wide"
)

st.title("⚡ Clinical Explainable EEG Seizure Dashboard (V3)")
st.caption(
    "محرك تشخيص إكلينيكي موحد ومستقل عن عدد القنوات (Universal Channel-Agnostic Engine v3.0)"
)

uploaded_file = st.file_uploader("رفع ملف رسم المخ (EDF Format)", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # أ) تحميل الملف
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    with raw.info._unlock():
        raw.info["subject_info"] = None

    # ب) استخراج الخصائص الموحدة بالكامل (18 ميزة)
    feats, psd_data, epochs = extract_v3_comprehensive_features(raw)

    # ج) التنبؤ بالنموذج
    feats_scaled = scaler.transform(feats)
    preds = model.predict(feats_scaled)
    probs_all = model.predict_proba(feats_scaled)
    probs = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]

    mean_prob = np.mean(probs)
    seizure_pct = np.mean(preds) * 100

    # د) تحديد القرار الإكلينيكي وتفعيل النطاق الشكوكي (Uncertainty Evaluation)
    if mean_prob > 0.60:
        diag_status = "POSITIVE - Seizure Activity Detected"
        st.error(
            f"🚨 **تنبيه إكلينيكي عاجل: تم اكتشاف نشاط صرعي (Seizure Detected)** | نسبة القطاعات المصابة: {seizure_pct:.1f}%"
        )
    elif 0.40 <= mean_prob <= 0.60:
        diag_status = "BORDERLINE - Clinical Review Required"
        st.warning(
            f"⚠️ **تنبيه الشك الإكلينيكي: حالة حدية مشكوك فيها (Indeterminate Pattern)** | نسبة القطاعات المصابة: {seizure_pct:.1f}% - يوصى بمراجعة الاستشاري يدوياً."
        )
    else:
        diag_status = "NEGATIVE - Normal EEG Pattern"
        st.success(
            f"✅ **نتيجة فحص سليمة: رسم مخ طبيعي (Normal EEG)** | نسبة القطاعات المصابة: {seizure_pct:.1f}%"
        )

    psd_summary = {
        "Delta (0.5-4 Hz)": float(feats[:, 4].mean()),
        "Theta (4-8 Hz)": float(feats[:, 5].mean()),
        "Alpha (8-13 Hz)": float(feats[:, 6].mean()),
        "Beta (13-30 Hz)": float(feats[:, 7].mean()),
    }

    # هـ) الرسم البصري المزدوج (التتبع الزمني + Topomap)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 التتبع الزمني لاحتمالية النوبة")
        fig_time, ax_time = plt.subplots(figsize=(7, 4))
        ax_time.plot(
            np.arange(len(probs)) * 2.0,
            probs,
            color="red" if mean_prob > 0.5 else "blue",
            linewidth=2,
        )
        ax_time.axhline(0.5, color="gray", linestyle="--")
        ax_time.set_title("Temporal Seizure Probability Profile")
        ax_time.set_xlabel("Time (Seconds)")
        ax_time.set_ylabel("Probability")

        tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig_time.savefig(tmp_plot.name, bbox_inches="tight")
        st.pyplot(fig_time)

    topomap_tmp_path = None
    with col2:
        st.subheader("📍 الخريطة المكانية لنشاط المخ (2D Topomap)")
        try:
            raw_topo = raw.copy()
            montage = mne.channels.make_standard_montage("standard_1020")
            raw_topo.set_montage(montage, on_missing="ignore")

            psd_mean_clean = psd_data.mean(axis=(0, 2))

            fig_topo, ax_topo = plt.subplots(figsize=(5.5, 4))
            mne.viz.plot_topomap(
                psd_mean_clean,
                raw_topo.info,
                axes=ax_topo,
                show=False,
                contours=6,
                sensors=True,
            )
            ax_topo.set_title("Spatial Power Spectral Density")

            tmp_topo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_topo.savefig(tmp_topo.name, bbox_inches="tight")
            topomap_tmp_path = tmp_topo.name

            st.pyplot(fig_topo)
        except Exception as e:
            st.info("معلومات: تم إيقاف الخريطة المكانية لعدم تطابق المونتاج.")

    # و) زر تنزيل التقرير الطبي PDF
    pdf_path = generate_clinical_pdf_report(
        diag_status,
        seizure_pct,
        len(preds),
        psd_summary,
        tmp_plot.name,
        topomap_tmp_path,
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 تنزيل التقرير الطبي المعتمد للمستشفى (PDF)",
            f,
            file_name=f"Clinical_EEG_Report_V3_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
