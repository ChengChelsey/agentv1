# detectors/base.py
from typing import List, Tuple, Optional, Dict, Any, Union

class DetectionResult:
    def __init__(
        self,
        method: str,
        anomalies: Optional[List[int]] = None,
        anomaly_scores: Optional[List[float]] = None,
        intervals: Optional[List[Tuple[int, int]]] = None,
        auxiliary_curve: Optional[List[Tuple[int, float]]] = None,
        description: str = "",
        visual_type: str = "point",  # point | range | curve | none
        explanation: Optional[List[str]] = None,
    ):
        self.method = method
        self.anomalies = anomalies or []
        self.anomaly_scores = anomaly_scores or []
        self.intervals = intervals or []
        self.auxiliary_curve = auxiliary_curve or []
        self.description = description
        self.visual_type = visual_type
        self.explanation = explanation or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "anomalies": self.anomalies,
            "anomaly_scores": self.anomaly_scores,
            "intervals": self.intervals,
            "auxiliary_curve": self.auxiliary_curve,
            "description": self.description,
            "visual_type": self.visual_type,
            "explanation": self.explanation
        }