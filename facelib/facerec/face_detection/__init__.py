"""Face detection."""
from .cascade import HAARCascade as HAARFaceDetector
from .cascade import LBPCascade as LBPFaceDetector
from .ssd_quantized import SSD as SSDFaceDetector