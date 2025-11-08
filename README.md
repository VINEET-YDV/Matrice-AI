# Matrice-AI
Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)

Eg of json format:
{
  "timestamp": 1713459200,
  "frame_id": 32,
  "stream_name": "cam_1",
  "latency_ms": 20.1,
  "detections": [
    {
      "label": "person",
      "conf": 0.88,
      "bbox": [120.5, 300.2, 250.0, 580.7]
    },
    {
      "label": "car",
      "conf": 0.61,
      "bbox": [400.0, 500.1, 480.2, 550.0]
    }
  ]
}
