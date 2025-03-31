# detectors/zscore.py
import numpy as np
from typing import List, Tuple, Optional
from detectors.base import DetectionResult

class ZScoreDetector:
    def __init__(self, threshold: float = 3.0):

        self.threshold = threshold
    
    def detect(self, series: List[Tuple[int, float]]) -> DetectionResult:

        if not series:
            return DetectionResult(
                method="Z-Score",
                description="无数据进行Z-Score分析",
                visual_type="none"
            )
        

        timestamps = [t for t, _ in series]
        values = np.array([v for _, v in series])

        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1.0
        
        z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
        
        anomalies = []
        scores = []
        explanations = []
        
        for i, z in enumerate(z_scores):
            if abs(z) > self.threshold:
                anomalies.append(timestamps[i])
                scores.append(float(abs(z)))
                # 解释是高于均值还是低于均值
                direction = "高于" if z > 0 else "低于"
                explanations.append(f"Z-Score={z:.2f}，{direction}均值{abs(z):.2f}个标准差")
        
        return DetectionResult(
            method="Z-Score",
            anomalies=anomalies,
            anomaly_scores=scores,
            description=f"使用Z-Score方法(阈值={self.threshold})检测到{len(anomalies)}个异常点",
            visual_type="point",
            explanation=explanations
        )