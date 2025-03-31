# analysis/single_series.py
import config
import numpy as np
import logging
from detectors.base import DetectionResult

logger = logging.getLogger("anomaly_detection.single_series")

def analyze_single_series(series):

    if not series:
        logger.warning("输入时间序列为空")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
        
    values = [v for _, v in series]
    if len(set(values)) <= 1:
        logger.info("输入时间序列几乎不变，无需检测")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
        
    from detectors.zscore import ZScoreDetector
    from detectors.cusum import CUSUMDetector
    
    thres = config.THRESHOLD_CONFIG
    try:
        res_z = ZScoreDetector(
            threshold=thres.get("Z-Score", {}).get("threshold", 3.5)
        ).detect(series)
        logger.info(f"Z-Score 检测到 {len(res_z.anomalies)} 个异常点")
    except Exception as e:
        logger.error(f"Z-Score 检测失败: {e}")
        res_z = DetectionResult(method="Z-Score", description=f"检测失败: {e}")

    try:
        res_cusum = CUSUMDetector(
            drift_threshold=thres.get("CUSUM", {}).get("drift_threshold", 6.0),
            k=thres.get("CUSUM", {}).get("k", 0.7)
        ).detect(series)
        logger.info(f"CUSUM 检测到 {len(res_cusum.anomalies)} 个异常点, {len(res_cusum.intervals)} 个异常区间")
    except Exception as e:
        logger.error(f"CUSUM 检测失败: {e}")
        res_cusum = DetectionResult(method="CUSUM", description=f"检测失败: {e}")

    valid_results = []
    for result in [res_z, res_cusum]:
        if len(result.anomalies) > 0 or len(result.intervals) > 0 or result.visual_type != "none":
            valid_results.append(result)
    
    if not valid_results:
        logger.warning("所有检测方法都未找到异常或失败")
        return {
            "method_results": [res_z, res_cusum],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
    
    method_results = []
    method_names = set()
    
    for result in [res_z, res_cusum]:
        if result.method in method_names:
            logger.info(f"跳过重复的 {result.method} 结果")
            continue
        method_results.append(result)
        method_names.add(result.method)
    
    total_weight = 0.0
    composite_score = 0.0
    length = len(series)
    
    for res in method_results:
        m_name = res.method
        weight = config.WEIGHTS_SINGLE.get(m_name, 0.3)
        total_weight += weight
        
        anomalies_count = len(res.anomalies)
        intervals_count = len(res.intervals) * 3
        total_count = anomalies_count + intervals_count
        
        if res.visual_type == "none":
            method_score = 0
        elif total_count > 0:
            if total_count / length < 0.01:
                method_score = 0.2 + 0.3 * (total_count / length) * 100
            else:
                ratio = total_count / length
                method_score = min(0.9, 0.2 + 0.3 * np.log10(1 + ratio * 100))
        else:
            method_score = 0
        
        logger.info(f"方法 {m_name} 得分: {method_score:.2f}, 权重: {weight}")
        composite_score += weight * method_score
    
    if total_weight > 0:
        composite_score /= total_weight
    
    methods_with_anomalies = sum(1 for res in method_results 
                              if len(res.anomalies) > 0 or len(res.intervals) > 0)
    if methods_with_anomalies == 1 and len(method_results) > 1:
        logger.info("仅一个方法检测到异常，降低得分")
        composite_score *= 0.8
    
    classification = (
        "高置信度异常" if composite_score >= config.HIGH_ANOMALY_THRESHOLD
        else "轻度异常" if composite_score >= config.MILD_ANOMALY_THRESHOLD
        else "正常"
    )
    
    logger.info(f"综合得分: {composite_score:.2f}, 分类: {classification}")
    
    #合并所有异常点
    all_anomalies = set()
    for r in method_results:
        all_anomalies.update(r.anomalies)

    anomaly_ratio = len(all_anomalies) / length if length > 0 else 0

    if anomaly_ratio > 0.25:
        logger.warning(f"异常点比例高达 {anomaly_ratio:.1%}，重新评估分类")
        if methods_with_anomalies < len(method_results) * 0.7: 
            if classification == "高置信度异常":
                classification = "轻度异常"
                logger.info("降级为轻度异常")
            elif classification == "轻度异常" and anomaly_ratio > 0.4:
                classification = "正常"
                logger.info("降级为正常")
    
    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": sorted(all_anomalies)
    }