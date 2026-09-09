import streamlit as st
import joblib, mne, numpy as np
import matplotlib.pyplot as plt

st.title("Clinical EEG Seizure Dashboard")

uploaded_file = st.file_uploader("ارفع ملف رسم المخ (EDF)", type=["edf"])

if uploaded_file is not None:
    try:
        pipe = joblib.load('eeg_pipeline.joblib')
        model, scaler = pipe['model'], pipe['scaler']
        
        raw = mne.io.read_raw_edf(uploaded_file.name, preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        raw.resample(250, verbose=False)
        
        data = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False).get_data()
        feats = np.hstack((data.mean(axis=-1), data.std(axis=-1)))
        feats_scaled = scaler.transform(feats)
        
        preds = model.predict(feats_scaled)
        probs = model.predict_proba(feats_scaled)[:, 1]
        
        is_seizure = np.mean(preds) > 0.5
        st.subheader("التقرير التشخيصي")
        if is_seizure:
            st.error("⚠️ نشاط صرعي محتمل (Seizure Detected)")
        else:
            st.success("✅ رسم مخ طبيعي (Normal EEG)")
            
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.arange(len(probs)) * 2.0, probs, color='red', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.set_title('Temporal Seizure Probability Profile')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
