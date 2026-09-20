import os
import re
import tempfile
from datetime import datetime
from fpdf import FPDF
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# 1. تحميل النموذج والـ Scaler
pipe = joblib.load("eeg_pipeline_v2.joblib")
model, scaler = pipe["model"], pipe["scaler"]


# 2. إنشاء التقرير الطبي المعتمد للمستشفيات (EHR-Ready PDF Report)
def generate_clinical_pdf_report(
    is_seizure_flag,
    seizure_pct,
    num_epochs,
    psd_summary,
    plot_path,
    topomap_path=None,
):
    pdf = FPDF()
    pdf.add_page()

    # الهيدر الرسمي للمستشفى / المركز الطبي
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(
        200, 10, txt="CLINICAL EEG AUTOMATED DIAGNOSTIC REPORT", ln=True, align="C"
    )
    pdf.set_font("Helvetica", size=9)
    pdf.cell(
        200,
        5,
        txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Accession ID: EEG-AI-{np.random.randint(10000, 99999)}",
        ln=True,
        align="C",
    )
    pdf.ln(5)

    # شريط نتيجة التشخيص الإكلينيكي
    pdf.set_font("Helvetica", size=12, style="B")
    diag_status = (
        "POSITIVE - Seizure Activity Detected"
        if is_seizure_flag
        else "NEGATIVE - Normal EEG Pattern"
    )
    pdf.cell(200, 8, txt=f"Primary Diagnostic Finding: {diag_status}", ln=True)

    # تفاصيل التحليل الإحصائي
    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        200, 6, txt=f"Seizure Burden (Epochs Ratio): {seizure_pct:.1f}%", ln=True
    )
    pdf.cell(
        200, 6, txt=f"Total Analyzed Epochs (2-sec Windows): {num_epochs}", ln=True
    )
    pdf.ln(4)

    # توزيع طاقات التردد spectral power
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(
        200, 8, txt="Spectral Power Band Distribution (Mean PSD):", ln=True
    )
    pdf.set_font("Helvetica", size=10)
    for band, val in psd_summary.items():
        pdf.cell(200, 5, txt=f" - {band}: {val:.4f} uV^2/Hz", ln=True)
    pdf.ln(5)

    # إدراج رسم التتبع الزمني
    if os.path.exists(plot_path):
        pdf.set_font("Helvetica", size=11, style="B")
        pdf.cell(
            200, 8, txt="Temporal Probability Profile Across Time:", ln=True
        )
        pdf.image(plot_path, x=15, w=180)
        pdf.ln(4)

    # إدراج خريطة الجمجمة الحرارية 2D Topomap
    if topomap_path and os.path.exists(topomap_path):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(
            200,
            10,
            txt="2D Spatial Localization & Topographic Power Heatmap:",
            ln=True,
        )
        pdf.image(topomap_path, x=25, w=160)
        pdf.ln(10)

    # خانة توقيع الطبيب المعالج والاعتماد الإكلينيكي
    pdf.ln(10)
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(100, 6, txt="Attending Physician Signature: __________________", ln=False)
    pdf.cell(90, 6, txt="Date: _____________", ln=True)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# 3. واجهة التطبيق عبر Streamlit
st.set_page_config(
    page_title="Clinical EEG Seizure Dashboard", layout="wide"
)

st.title("⚡ Clinical Explainable EEG Seizure Dashboard")
st.write(
    "نظام تشخيص إكلينيكي محكّم ومباشر (GroupKFold + Automated Artifact Removal + Universal Channel Mapping)"
)

uploaded_file = st.file_uploader("رفع ملف رسم المخ (EDF Format)", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # أ) تحميل الملف وإزالة هوية المريض
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    with raw.info._unlock():
        raw.info["subject_info"] = None

    # ب) تنظيف وتوحيد أسماء القنوات وإزالة أي قنوات مكررة
    mapping = {}
    seen = set()
    channels_to_drop = []

    for ch in raw.ch_names:
        clean_name = re.sub(r"(EEG|Ref|-Ref|-A1|-A2|\.)", "", ch, flags=re.IGNORECASE).strip()
        if clean_name in seen or clean_name == "":
            channels_to_drop.append(ch)
        else:
            mapping[ch] = clean_name
            seen.add(clean_name)

    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    raw.rename_channels(mapping)

    # ج) معالجة وتصفية الإشارة من التشويش
    raw.filter(0.5, 40.0, verbose=False)
    raw.resample(250, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    data = epochs.get_data()

    # د) استخراج الخصائص الزمنية والترددية
    mean_feat = data.mean(axis=-1)
    std_feat = data.std(axis=-1)

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

    # هـ) التنبؤ بالنموذج المعتمد
    feats_scaled = scaler.transform(feats)
    preds = model.predict(feats_scaled)
    probs_all = model.predict_proba(feats_scaled)
    probs = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]

    seizure_pct = np.mean(preds) * 100
    is_seizure = np.mean(preds) > 0.5

    psd_summary = {
        "Delta (0.5-4 Hz)": float(delta.mean()),
        "Theta (4-8 Hz)": float(theta.mean()),
        "Alpha (8-13 Hz)": float(alpha.mean()),
        "Beta (13-30 Hz)": float(beta.mean()),
    }

    # و) عرض التنبيه الإكلينيكي
    if is_seizure:
        st.error(
            f"🚨 **تنبيه إكلينيكي عاجل: تم اكتشاف نشاط صرعي (Seizure Detected)** | نسبة القطاعات المصابة: {seizure_pct:.1f}%"
        )
    else:
        st.success(
            f"✅ **نتيجة فحص سليمة: رسم مخ طبيعي (Normal EEG)** | نسبة القطاعات المصابة: {seizure_pct:.1f}%"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 التتبع الزمني لاحتمالية النوبة")
        fig_time, ax_time = plt.subplots(figsize=(7, 4))
        ax_time.plot(
            np.arange(len(probs)) * 2.0,
            probs,
            color="red" if is_seizure else "blue",
            linewidth=2,
        )
        ax_time.axhline(0.5, color="gray", linestyle="--")
        ax_time.set_title("Temporal Seizure Probability Profile")
        ax_time.set_xlabel("Time (Seconds)")
        ax_time.set_ylabel("Probability")

        tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig_time.savefig(tmp_plot.name, bbox_inches="tight")
        st.pyplot(fig_time)

    # ز) إنشاء خريطة الجمجمة الحرارية (2D Topomap) مع تصفية التداخل المباشر
    topomap_tmp_path = None
    with col2:
        st.subheader("📍 الخريطة المكانية لنشاط المخ (2D Topomap)")
        try:
            raw_topo = raw.copy()
            montage = mne.channels.make_standard_montage("standard_1020")
            raw_topo.set_montage(montage, on_missing="ignore")

            # إزالة القنوات المسببة لتداخل الإحداثيات عند الرسم
            overlapping_bad_channels = [
                'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 
                'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 
                'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'Ft7', 'Ft8', 
                'Tp7', 'Tp8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8'
            ]
            
            ch_to_drop = [ch for ch in overlapping_bad_channels if ch in raw_topo.ch_names]
            if ch_to_drop:
                raw_topo.drop_channels(ch_to_drop)

            valid_indices = [raw.ch_names.index(ch) for ch in raw_topo.ch_names]
            psd_mean_clean = psd_data[:, valid_indices, :].mean(axis=(0, 2))

            fig_topo, ax_topo = plt.subplots(figsize=(5.5, 4))
            
            mne.viz.plot_topomap(
                psd_mean_clean,
                raw_topo.info,
                axes=ax_topo,
                show=False,
                contours=6,
                sensors=True
            )
            ax_topo.set_title("Spatial Power Spectral Density")

            tmp_topo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_topo.savefig(tmp_topo.name, bbox_inches="tight")
            topomap_tmp_path = tmp_topo.name

            st.pyplot(fig_topo)
        except Exception as e:
            st.warning(f"تنبيه تقني: تعذر معالجة الخريطة: {e}")

    # ح) توليد زر تحميل التقرير الطبي المعتمد
    pdf_path = generate_clinical_pdf_report(
        is_seizure,
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
            file_name=f"Clinical_EEG_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
