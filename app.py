import streamlit as st
import joblib, mne, numpy as np, tempfile, os
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="Clinical EEG Seizure Dashboard", layout="wide")

st.title("⚡ Clinical EEG Seizure & XAI Dashboard")
st.markdown("نظام الذكاء الاصطناعي المتقدم لتشخيص الصرع، تحديد البؤرة الكهربائية، واستخراج التقارير الطبية.")

# دالة إنشاء تقرير PDF
def create_pdf_report(is_seizure, avg_prob, duration_sec):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Clinical EEG Diagnostic Report", ln=True, align='C')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    
    status = "Seizure Activity Detected" if is_seizure else "Normal EEG Pattern"
    pdf.cell(0, 10, f"Diagnostic Finding: {status}", ln=True)
    pdf.cell(0, 10, f"Average Seizure Probability: {avg_prob*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Record Duration Analyzed: {duration_sec:.1f} Seconds", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Generated automatically by Clinical EEG AI Engine.", ln=True)
    
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

uploaded_file = st.file_uploader("ارفع ملف رسم المخ (EDF)", type=["edf"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        pipe = joblib.load('eeg_pipeline.joblib')
        model, scaler = pipe['model'], pipe['scaler']
        
        # 1. القراءة والتعديل التلقائي
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        raw.resample(250, verbose=False)
        
        # 2. إعداد الخريطة والأسماء
        raw.rename_channels({ch: ch.replace('.', '').strip() for ch in raw.ch_names})
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch in raw.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # 3. استخراج الخصائص والتنبؤ
        data = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False).get_data()
        feats = np.hstack((data.mean(axis=-1), data.std(axis=-1)))
        feats_scaled = scaler.transform(feats)
        
        preds = model.predict(feats_scaled)
        probs = model.predict_proba(feats_scaled)[:, 1]
        
        avg_prob = np.mean(probs)
        is_seizure = avg_prob > 0.5
        duration_sec = raw.times[-1]
        
        # العرض الإكلينيكي
        st.subheader("📋 التقرير التشخيصي الإكلينيكي")
        col1, col2 = st.columns(2)
        
        with col1:
            if is_seizure:
                st.error("⚠️ نشاط صرعي محتمل (Seizure Activity Detected)")
            else:
                st.success("✅ رسم مخ طبيعي (Normal EEG Pattern)")
        
        with col2:
            st.metric(label="متوسط احتمالية الصرع", value=f"{avg_prob*100:.1f}%")

        # زر تحميل التقرير PDF
        pdf_path = create_pdf_report(is_seizure, avg_prob, duration_sec)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 تحميل التقرير الطبي (PDF Report)",
                data=f,
                file_name="EEG_Clinical_Report.pdf",
                mime="application/pdf"
            )

        st.markdown("---")
        
        # التبويبات المتعددة
        tab1, tab2 = st.tabs(["📈 التتبع الزمني (Temporal Tracking)", "🧠 الخريطة الحرارية للبؤرة (XAI Topomap)"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            times = np.arange(len(probs)) * 2.0
            ax.plot(times, probs, color='#d9534f', linewidth=2, label='Seizure Probability')
            ax.axhline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
            ax.fill_between(times, probs, 0.5, where=(probs >= 0.5), color='red', alpha=0.3)
            ax.set_xlabel('Time (Seconds)')
            ax.set_ylabel('Probability')
            ax.set_title('Temporal Seizure Probability Profile (2-sec Windows)')
            ax.legend()
            st.pyplot(fig)
            
        with tab2:
            fig_topo, ax_topo = plt.subplots(figsize=(6, 6))
            importances = getattr(model, 'feature_importances_', np.ones(feats.shape[1]))
            channel_imp = importances[:len(raw.ch_names)]
            
            im, _ = mne.viz.plot_topomap(channel_imp, raw.info, axes=ax_topo, show=False)
            fig_topo.colorbar(im, ax=ax_topo, orientation='vertical', shrink=0.8, label='Biomarker Importance')
            ax_topo.set_title("XAI: Topomap of Seizure Focus Importance", fontweight='bold')
            st.pyplot(fig_topo)
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
