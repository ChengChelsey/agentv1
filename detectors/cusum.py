# detectors/cusum.py
import numpy as np
from typing import List, Tuple, Optional
from detectors.base import DetectionResult

class CUSUMDetector:
    def __init__(self, drift_threshold: float = 5.0, k: float = 0.5):
        """
        参数:
            drift_threshold: CUSUM阈值
            k: 灵敏度参数，较小的值对小偏移更敏感
        """
        self.drift_threshold = drift_threshold
        self.k = k
    
    def detect(self, series: List[Tuple[int, float]]) -> DetectionResult:
        if not series:
            return DetectionResult(
                method="CUSUM",
                description="无数据进行CUSUM分析",
                visual_type="none"
            )
        
        timestamps = [t for t, _ in series]
        values = np.array([v for _, v in series])
        
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1.0
        
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        

        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mean)/std - self.k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mean)/std - self.k)
        
        cusum_combined = np.maximum(cusum_pos, cusum_neg)
        
        anomalies = []
        scores = []
        for i, c in enumerate(cusum_combined):
            if c > self.drift_threshold:
                anomalies.append(timestamps[i])
                scores.append(float(c))
        
        explanations = [
            f"CUSUM值={scores[i]:.2f}，累计偏移超过阈值({self.drift_threshold})"
            for i in range(len(anomalies))
        ]
        
        from analysis.multi_series import group_anomaly_times
        intervals = group_anomaly_times(anomalies)
        
        cum_curve = [(timestamps[i], float(cusum_combined[i])) for i in range(len(timestamps))]
        
        return DetectionResult(
            method="CUSUM",
            anomalies=anomalies,
            anomaly_scores=scores,
            intervals=intervals,
            auxiliary_curve=cum_curve,
            description=f"CUSUM累积偏移检测到 {len(intervals)} 个异常区段，共 {len(anomalies)} 个高偏移点",
            visual_type="curve",
            explanation=explanations
        )