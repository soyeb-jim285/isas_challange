
## 5 Discussion

This section interprets the experimental findings with respect to three core dimensions—overall activity‐level performance, error structure revealed by the confusion matrix, and per‑activity outcomes—while situating the results within contemporary human‑activity‑recognition (HAR) scholarship.

### 5.1 Overall Activity‑Level Results

The optimized LSTM raised weighted F1‑score and accuracy by 1.3 % and 1.5 %, respectively, relative to the baseline (Table 2). Although incremental in absolute terms, such gains are noteworthy given the severe class imbalance (unusual behaviours constitute <25 % of frames) and the small‑sample, leave‑one‑subject‑out (LOSO) evaluation. Prior skeleton‑based works on abnormal HAR—e.g., Morais et al. [5] and Yan et al.’s ST‑GCN [6]—report similar magnitudes of improvement when architectural refinements are introduced, suggesting that the observed uplift is both credible and practically meaningful. Importantly, variance across LOSO folds fell by roughly one‑third (σ_Acc: 0.051 → 0.035), indicating that the 90‑frame window, bidirectional memory, and aggressive dropout collectively strengthen generalisation to unseen subjects—an essential property for real‑world deployment in care homes where per‑resident fine‑tuning is impractical.

The three‑fold increase in window length, identified via grid search, corroborates the window‑size sensitivity documented in EEG‑based LSTM studies [4] and in video anomaly detection using sliding‑window CNN‑RNN hybrids. Longer windows evidently capture the preparatory and follow‑through phases typical of violent or self‑injurious acts, providing richer temporal context without materially degrading convergence speed (Figure 4).

### 5.2 Confusion Matrix and Misclassification Patterns

Figure 1 illuminates two persistent error modes. **First, high confusion among sedentary normal activities** (“Using phone,” “Sitting quietly,” “Eating snacks”) arises because skeletal key‑points differ primarily in subtle wrist and elbow trajectories that are (i) sometimes occluded and (ii) poorly represented by global‑motion features. This mirrors the pose‑only limitations reported in office‑environment datasets, where sitting‑at‑desk tasks are frequently entangled unless object cues are added. Future multi‑modal extensions—e.g., RGB object detection or inertial sensing—could disambiguate these cases.

**Second, residual overlap between vigorous social gestures and aggressive acts** can be seen in the off‑diagonal “Attacking” cells. While head‑banging and throwing events now form tighter clusters—thanks to the 18‑feature set’s inclusion of head and limb‑span metrics—the model occasionally conflates broad, expressive arm movements with assault behaviour when contextual cues (e.g., proximity to another person) are absent. Comparable false‑positives are reported by ST‑GCN models trained on the Kinetics‑Skeleton anomaly subset, reinforcing that pose alone is sometimes insufficient for intent discrimination.

Encouragingly, false‑negative rates for the three most safety‑critical classes—“Attacking,” “Head banging,” “Throwing things”—all declined, meaning the system is less likely to miss genuinely dangerous episodes, even at the cost of sporadic false alarms. In care‑facility settings, such a trade‑off is often preferred, as undetected harm imposes higher clinical and legal risks than occasional caregiver validation of benign events.

### 5.3 Per‑Activity Results and Implications

Table 4 and Figure 2 furnish granular insights:

- **Exceptional performers.** “Walking” retained near‑ceiling F1 (0.94) across folds—consistent with prior gait‑centric studies—confirming that cyclical, high‑energy motions generate distinctive temporal signatures easily modelled by LSTM cells.
    
- **Most improved.** “Throwing things” leapt by 14.3 % F1 after optimisation, illustrating that the 90‑frame context effectively spans the wind‑up, release, and follow‑through micro‑phases absent in the 30‑frame baseline. Similar findings were reported in Morais et al. [5], where extended temporal receptive fields boosted anomaly recall.
    
- **Persistently difficult.** “Using phone” (0.35 F1) and “Eating snacks” (0.39 F1) remain challenging due to (i) minimal gross‑motor change, (ii) inter‑subject variability in hand‑to‑mouth trajectories, and (iii) seated‑pose similarity. Attempts to enlarge the window further (>120 frames) delivered diminishing returns, echoing EEG‑window studies [4] which show that excessively long sequences dilute signal with idle frames.
    
- **Clinical relevance.** From a caregiving perspective, reliable detection of high‑risk acts (attacking, head‑banging, throwing) is paramount, whereas false‑positives on benign seated activities primarily affect caregiver workload. Accordingly, application‑layer thresholds could be lowered selectively for high‑severity classes or augmented with secondary verification (e.g., audio of impact sounds) to balance sensitivity and precision.