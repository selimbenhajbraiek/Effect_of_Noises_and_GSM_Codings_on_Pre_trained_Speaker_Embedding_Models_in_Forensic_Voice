## Effect of Noises and GSM Codings on Pre-trained Speaker Embedding Models in Forensic Voice Comparison

## Overview

This project explores the impact of noise and GSM coding on forensic voice comparison (FVC) performance. It utilizes deep speaker embedding models to analyze speaker verification under degraded conditions. The research evaluates speaker recognition using likelihood-ratio (LR) frameworks and key metrics such as Equal Error Rate (EER) and Log-Likelihood Ratio Cost (Cllr). The study aims to aid forensic experts in understanding how environmental and technical factors affect speaker verification systems.

## Features

Speaker Embedding Models: Utilizes four pre-trained models: X-vector, ECAPA-TDNN, Wav2Vec, and WavLM.

## Dataset: Uses the ForVoice 120+ Hungarian speech dataset.

Noise and GSM Impact: Evaluates white noise at SNRs of 10 dB and 15 dB, reverberation, and different GSM encoding rates.

Likelihood-Ratio Framework: Uses logistic regression for scoring.

Evaluation Metrics: Assesses performance using EER and Cllr.

## Methodology

Data Preparation:

Organized dataset with male and female speaker classification.

Extracted speaker embeddings using pre-trained models.

Noise and GSM Encoding Simulation:

Applied noise and encoding variations to dataset.

Speaker Verification:

Compared speaker pairs to evaluate verification accuracy.

Used cosine similarity and logistic regression for likelihood ratio scoring.

## Evaluation:

Computed EER and Cllr to measure model robustness.

Analyzed the effects of noise and GSM degradation.

## Results Summary

The ECAPA-TDNN model achieved the best performance with an EER of 0.01 on clean trials.

Performance degraded with increased noise and GSM encoding.

Mismatched channels (e.g., high-quality vs. low-bitrate GSM) significantly impacted accuracy.

## Tools & Dependencies

Python

PyTorch

SciKit-Learn

SpeechBrain

Pandas, NumPy

Librosa (for audio processing)
