import os
import tempfile
from fpdf import FPDF
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# 1. تحميل النموذج والـ Scaler
pipe = joblib.load("eeg_pipeline_v2.joblib")
model, scaler = pipe["model"], pipe["scaler"]


# 2. دالة إنتاج التقرير الطبي الشامل بصيغة PDF (تتضمن الصور والرسومات البيانية)
def generate_advanced_pdf_report(
    is_seizure_flag,
    seizure_pct,
    num_epochs,
    psd_summary,
    plot_path,
    topomap_path=None,
):
    pdf = FPDF()
    pdf.add_page()

    # العنوان الرئيسي
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(
        200, 10, txt="Clinical EEG Comprehensive Report", ln=True, align="C"
    )
    pdf.ln(5)

    # نتيجة التشخيص
    pdf.set_font("Helvetica", size=12, style="B")
    diag_status = (
        "POSITIVE (Seizure Activity Detected)"
        if is_seizure_flag
        else "NEGATIVE (Normal EEG Pattern)"
    )
    pdf.cell(200, 8, txt=f"Diagnostic Result: {diag_status}", ln=True)

    # التفاصيل والإحصائيات
    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        200, 6, txt=f"Seizure Epochs Percentage: {seizure_pct:.1f}%", ln=True
    )
    pdf.cell(200, 6, txt=f"Total Processed Epochs: {num_epochs}", ln=True)
    pdf.ln(4)

    # توزيع طاقات التردد
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(
        200, 8, txt="Spectral Power Band Distribution (PSD):", ln=True
    )
    pdf.set_font("Helvetica", size=10)
    for band, val in psd_summary.items():
        pdf.cell(200, 5, txt=f" - {band}: {val:.4f}", ln=True)
    pdf.ln(6)

    # إدراج رسم التتبع الزمني داخل الـ PDF
    if os.path.exists(plot_path):
        pdf.set_font("Helvetica", size=11, style="B")
        pdf.cell(
            200, 8, txt="Temporal Seizure Probability Profile:", ln=True
        )
        pdf.image(plot_path, x=15, w=180)
        pdf.ln(5)

    # إدراج خريطة الجمجمة الحرارية Topomap داخل الـ PDF
    if topomap_path and os.path.exists(topomap_path):
        pdf.add_page()  # صفحة جديدة للخريطة
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(
            200,
            10,
            txt="Spatial Localization & Brain Topography (2D Topomap):",
            ln=True,
        )
        pdf.image(topomap_path, x=25, w=160)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# 3. واجهة التطبيق عبر Streamlit
st.title("Clinical Explainable EEG Seizure Dashboard")
st.write(
    "نظام تشخيص إكلينيكي متكامل (GroupKFold + PSD Bands + 2D Spatial Localization)"
)

uploaded_file = st.file_uploader("رفع ملف EEG (EDF Format)", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # أ) معالجة وتصفية ملف EDF (حل مشكلة التباين والتشويش)
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    raw.filter(0.5, 40.0, verbose=False)  # تصفية التشويش
    raw.resample(250, verbose=False)  # توحيد معدل القراءات
    raw.rename_channels({ch: ch.replace(".", "").strip() for ch in raw.ch_names})

    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    data = epochs.get_data()

    # ب) استخراج الخصائص الزمنية
    mean_feat = data.mean(axis=-1)
    std_feat = data.std(axis=-1)

    # ج) استخراج نطاقات التردد PSD
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

    # د) التنبؤ بالنموذج المعاير
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

    # هـ) عرض النتيجة الرئيسية
    st.subheader(f"التشخيص النهائي: {diag_display}")
    st.write(f"نسبة القطاعات المصابة: {seizure_pct:.1f}%")

    # و) رسم وتوليد مخطط التتبع الزمني
    fig_time, ax_time = plt.subplots(figsize=(8, 3))
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

    # حفظ مخطط التتبع كمؤقت
    tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig_time.savefig(tmp_plot.name, bbox_inches="tight")
    st.pyplot(fig_time)

    # ز) إنشاء خريطة الجمجمة الحرارية (2D Topomap)
    topomap_tmp_path = None
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_topo = raw.copy().set_montage(montage, on_missing="ignore")

        fig_topo, ax_topo = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(
            psd_data.mean(axis=(0, 2)),
            raw_topo.info,
            axes=ax_topo,
            show=False,
        )
        ax_topo.set_title("Spatial Power Spectral Density (2D Topomap)")

        tmp_topo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig_topo.savefig(tmp_topo.name, bbox_inches="tight")
        topomap_tmp_path = tmp_topo.name

        st.subheader("📍 الخريطة المكانية لنشاط المخ (Spatial Localization)")
        st.pyplot(fig_topo)
    except Exception as e:
        st.info("ملاحظة: تعذر عرض خريطة Topomap لعدم تطابق أجهزة القنوات.")

    # ح) إنتاج وتحميل التقرير الشامل PDF
    pdf_path = generate_advanced_pdf_report(
        is_seizure,
        seizure_pct,
        len(preds),
        psd_summary,
        tmp_plot.name,
        topomap_tmp_path,
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 تنزيل التقرير الطبي الشامل والكامل (PDF)",
            f,
            file_name="Clinical_EEG_Comprehensive_Report.pdf",
        )
