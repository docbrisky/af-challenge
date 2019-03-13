# Deep learning based rhythm recognition from images of ECGs

Computerised electrocardiogram (ECG) analysis can be invaluable in the clinical setting. ECG data contains vital information regarding life-changing diagnoses, such as a heart attack or significant rhythm abnormalities. For those of us working in cardiology, ECG interpretation usually feels like second nature. But for clincians in other specialties, it can be a hard skill to keep up. An ECG with the computerised diagnosis of "atrial fibrillation" (AF) printed at the top can give a clinician that extra bit of confidence needed to start a patient on anticoagulant medication, which in turn can prevent a debilitating future stroke.

As diagnostic algorithms evolve, it can be useful for both clinicians and researchers to revisit old ECG data with improved tools. However, most computerised ECG analysers rely on raw signals, whereas the vast majority of stored ECG data takes the form of printed ECG traces. This repository contains the source code from a study where we explored the idea that deep learning based ECG analysers - by virtue of being more robust to incomplete and noisy data - might be able to process ECG data derived from images with a less marked loss of performance than previous systems. In fact, we found that for this particular rhythm recognition task (https://physionet.org/challenge/2017/), there was no loss of performance at all.

It may strike the attentive reader that we've approached this problem in a roundabout way. The pipeline for this experiment involves writing ECG data to image files (losing half the resolution in the process), preprocessing the image using greyscale compression and a modified Otsu filter, extrapolating the signal data (losing further resolution, introducing noise and losing absolute mV values in the process), and finally feeding the data into a convolutional neural network (CNN) modified to accept 2D signal data rather than 3D image data.  Why not simply fine-tune an off-the-shelf CNN (e.g. ResNet-50 pretrained on ImageNet) using the ECG images?

The answer lies in a quirk of this particular dataset. The AF challenge involved a four-label classification problem: normal sinus rhythm, AF, "other" or "too noisy to intepret". The competition organisers did not specify what criteria were used to define an ECG as belonging to the "other" category, but as we went through the process of exploratory data analysis it became clear that very transient abnormalities (e.g. a ventricular triplet) would trigger this classification. Some of the traces were well over a minute long, and the obvious first step in using them to retrain a CNN like ResNet-50 would have been splitting them into "bitesize" (e.g. one second) windows, both to boost the quantity of training data and to create image dimensions compatible with the network's convolution windows. However, the fact that a single, split-second abnormality could dictate the classification of the entire trace made this impossible without using a recurrent framework or some LSTM-type function, for which we didn't feel there were enough training samples (each record would then have consistuted a single training example, rather than being split into multiple, independent examples).

The advantage of being forced into the roundabout approach noted above is that, having extrapolated signal data (or some equivalent thereof) from images of ECGs, we can make use of leading-edge analysers that are designed to work with raw signal data without having to make substantive changes to their architecture. In the experiment for which this source code was used, we downloaded an open-source deep learning model that had been trained on raw data from the AF Challenge and shown to work well (https://github.com/fernandoandreotti/cinc-challenge2017), then simply fine-tuned it with our extrapolated signal data. This saved the early work of experimenting with optimal network architecture, tuning hyperparameters, working out how to avoid local optima, and so on. Ultimately, it allowed us to achieve a level of performance beyond what we might have managed starting from scratch on our own.

The study is being presented at the International Society for Computerized Electrocardiology 44th Annual Conference in Florida this April (2019, https://isce.site-ym.com/event/2019conference) and the full paper will be published later this year. In the meantime, we hope that this code may provide useful points for consideration for other research groups. The source code was written for research purposes, and as such may be less elegant than production-level code. However, it has been extensively annotated, and any further questions can be addressed to robbrisk@hotmail.com

All code and text is copyright Ulster University, 2019.
