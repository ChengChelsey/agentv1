# detectors/trend_drift_cusum.py
import numpy as np
from detectors.base import DetectionResult
from utils.time_utils import group_anomaly_times,format_timestamp

class TrendDriftCUSUMDetector:
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
    
    def detect(self, series1: list, series2: list) -> DetectionResult:
        if not series1 or not series2 or len(series1) < 10 or len(series2) < 10:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description="数据点不足进行趋势漂移分析(至少需要10个点)",
                visual_type="none"
            )
    
        residuals = []
        timestamps = [t for t, _ in series1]
        values1 = [v for _, v in series1]
        values2 = [v for _, v in series2]
        
        for (_, v1), (_, v2) in zip(series1, series2):
            residuals.append(v1 - v2)
        
        mean_abs_residual = np.mean(np.abs(residuals))
        max_abs_residual = np.max(np.abs(residuals))
        relative_diff = mean_abs_residual / (np.mean(np.abs(values1)) + 1e-10)

        if max_abs_residual < 0.05 and mean_abs_residual < 0.01:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"两序列几乎相同，最大差异仅{max_abs_residual:.3f}，平均差异{mean_abs_residual:.3f}",
                visual_type="none"
            )
        
        if relative_diff < 0.1:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"两序列差异不显著(相对差异{relative_diff:.1%})，无需进行漂移分析",
                visual_type="none"
            )
        
        def smooth_data(data, window=3):
            if len(data) < window:
                return data
            smoothed = np.convolve(data, np.ones(window)/window, mode='same')
            smoothed[:window//2] = data[:window//2]
            smoothed[-window//2:] = data[-window//2:]
            return smoothed
        
        smoothed_residuals = smooth_data(residuals, window=5)

        mean = np.mean(smoothed_residuals)
        std = np.std(smoothed_residuals) 
 
        if std < 1e-10:
            std = 1.0

        norm_residuals = [(r - mean) / std for r in smoothed_residuals]

        control_factor = 1.0
        cum_sum_pos = [0]
        cum_sum_neg = [0]
        
        for r in norm_residuals:
            cum_sum_pos.append(max(0, cum_sum_pos[-1] + r - control_factor))
            cum_sum_neg.append(max(0, cum_sum_neg[-1] - r - control_factor))
        
        cum_sum_pos = cum_sum_pos[1:]
        cum_sum_neg = cum_sum_neg[1:]
        cum_sum = [max(p, n) for p, n in zip(cum_sum_pos, cum_sum_neg)]

        anomalies = []
        scores = []
        consecutive_count = 0
        required_consecutive = 3 #至少需要连续3个点超过阈值
        
        for i, c in enumerate(cum_sum):
            if c > self.threshold:
                consecutive_count += 1
                if consecutive_count >= required_consecutive:
    
                    if consecutive_count == required_consecutive:
                        anomalies.append(timestamps[i - required_consecutive + 1])
                        scores.append(float(c))
                    anomalies.append(timestamps[i])
                    scores.append(float(c))
            else:
                consecutive_count = 0
        
        intervals = group_anomaly_times(anomalies, max_gap=300) #5分钟间隔

        filtered_intervals = []
        explanations = []
        
        for interval in intervals:
            start, end = interval
            duration = end - start
            
            interval_indices = [i for i, ts in enumerate(timestamps) if start <= ts <= end]
            if not interval_indices:
                continue
                
            interval_scores = [cum_sum[i] for i in interval_indices]
            avg_score = np.mean(interval_scores) if interval_scores else 0
            max_score = np.max(interval_scores) if interval_scores else 0
                
 
            if duration >= 300 and avg_score > self.threshold * 1.3 and max_score > self.threshold * 1.5:
                filtered_intervals.append(interval)
                explanations.append(
                    f"区间{format_timestamp(start)}至{format_timestamp(end)}的CUSUM值平均为{avg_score:.1f}，最大值{max_score:.1f}，超过阈值{self.threshold}，表明两序列存在持续趋势差异"
                )
        
        if not filtered_intervals:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"趋势漂移检测未发现明显的持续性异常区段",
                visual_type="none"
            )
        
        #辅助曲线数据
        aux_curve = [(timestamps[i], float(cum_sum[i])) for i in range(len(timestamps))]
        
        return DetectionResult(
            method="TrendDriftCUSUM",
            anomalies=[],
            anomaly_scores=[],
            intervals=filtered_intervals,
            auxiliary_curve=aux_curve,
            description=f"趋势漂移检测发现 {len(filtered_intervals)} 个明显异常区段，相对差异{relative_diff:.1%}",
            visual_type="range",
            explanation=explanations
        )

    