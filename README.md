# Deep learning based rhythm recognition from images of ECGs

Computerised electrocardiogram (ECG) analysis can be invaluable in the clinical setting. ECG data contains vital information regarding life-changing diagnoses, such as a heart attack or significant rhythm abnormalities. For those of us working in cardiology, ECG interpretation usually feels like second nature. But for clincians in other specialties, it can be a hard skill to keep up. An ECG with the computerised diagnosis of "atrial fibrillation" (AF) printed at the top can give a clinician that extra bit of confidence needed to start a patient on anticoagulant medication, which in turn can prevent a debilitating future stroke.

As diagnostic algorithms evolve, it can be useful for both clinicians and researchers to revisit old ECG data with improved tools. However, most computerised ECG analysers rely on raw signals, whereas the vast majority of stored ECG data takes the form of printed ECG traces. This repository contains the source code from a study where we explored the idea that deep learning based ECG analysers - by virtue of being more robust to incomplete and noisy data - might be able to process ECG data derived from images with a less marked loss of performance than previous systems. In fact, we found that for this particular rhythm recognition task (https://physionet.org/challenge/2017/), there was no loss of performance at all.

The source code was written for research purposes, and as such may be less elegant than production-level code. In particular, it is minimally modular. However, it has been extensively annotated, and any further questions can be addressed to robbrisk@hotmail.com

All code and text is copyright Ulster University, 2019.
