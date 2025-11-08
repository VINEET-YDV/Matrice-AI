"""
High-Performance YOLOv8 Inference Server
Version: Formatted Output

This server receives frames via WebSocket, performs inference using a multi-threaded
pipeline, and sends back JSON results in a structured, standardized format.
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
import time
import threading
import queue
import os
import onnxruntime as ort

# --- Configuration ---
MODEL_PATH = "yolov8n.onnx"
INPUT_SHAPE = (640, 640)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_QUEUE_SIZE = 10
PROCESSING_THREADS = 2 # Number of threads for pre/post-processing
INFERENCE_THREADS = 1  # Number of threads for model inference

# YOLOv8 class names
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# --- Helper: Performance Timer ---
class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        print(f"[PERF] {self.name}: {elapsed:.2f} ms")

# --- 1. Frame Receiver & Pre-processing Thread ---
def frame_receiver(frame_queue, raw_frame_queue, stop_event):
    """
    Receives raw (JPEG) frames from clients and puts them in a queue.
    """
    while not stop_event.is_set():
        try:
            # Get raw frame from the shared queue
            client_id, stream_name, frame_id, raw_data, receive_time = raw_frame_queue.get(timeout=1.0)
            
            with Timer(f"{stream_name} (Frame {frame_id}) - Pre-processing"):
                # 1. Decode JPEG
                frame = cv2.imdecode(np.frombuffer(raw_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"[WARN] {stream_name}: Failed to decode frame {frame_id}.")
                    continue
                
                original_shape = frame.shape[:2] # (height, width)

                # 2. Letterbox/Resize
                image_height, image_width = original_shape
                input_height, input_width = INPUT_SHAPE

                scale = min(input_width / image_width, input_height / image_height)
                scaled_width = int(image_width * scale)
                scaled_height = int(image_height * scale)
                
                scaled_image = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                
                canvas = np.full((input_height, input_width, 3), 0, dtype=np.uint8)
                
                x_offset = (input_width - scaled_width) // 2
                y_offset = (input_height - scaled_height) // 2
                
                canvas[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = scaled_image
                
                # 3. Normalize and Transpose
                input_tensor = canvas.astype(np.float32) / 255.0
                input_tensor = np.transpose(input_tensor, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0) # (1, C, H, W)

            # Put pre-processed frame into the inference queue
            if not frame_queue.full():
                # Pass all metadata through
                frame_queue.put((client_id, stream_name, frame_id, input_tensor, 
                                 original_shape, scale, x_offset, y_offset, receive_time), 
                                 timeout=1.0)
            else:
                print(f"[WARN] {stream_name}: Inference queue is full. Dropping frame {frame_id}.")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Pre-processing thread: {e}")

# --- 2. Inference Thread ---
def inference_worker(frame_queue, result_queue, stop_event, model_path):
    """
    Worker thread that performs model inference.
    """
    print("[INFO] Initializing ONNXRuntime session...")
    
    available_providers = ort.get_available_providers()
    print(f"[DEBUG] Available ONNX providers: {available_providers}")

    provider = "CPUExecutionProvider"
    if "CUDAExecutionProvider" in available_providers:
        provider = "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in available_providers:
        provider = "DmlExecutionProvider"
        
    print(f"[INFO] Using ONNX provider: {provider}")
    
    if provider == "CPUExecutionProvider":
        print("[WARNING] ****************************************************")
        print("[WARNING] * GPU NOT DETECTED! Falling back to CPU.          *")
        print("[WARNING] * Performance will be VERY SLOW.                  *")
        print("[WARNING] ****************************************************")

    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        print("[INFO] ONNXRuntime session initialized.")
    except Exception as e:
        print(f"[FATAL] Failed to create InferenceSession: {e}")
        stop_event.set()
        return

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    while not stop_event.is_set():
        try:
            # Get data from pre-processing
            (client_id, stream_name, frame_id, input_tensor, 
             original_shape, scale, x_offset, y_offset, receive_time) = frame_queue.get(timeout=1.0)
            
            with Timer(f"{stream_name} (Frame {frame_id}) - Inference"):
                outputs = session.run([output_name], {input_name: input_tensor})
            
            # Pass results and all metadata to post-processing
            if not result_queue.full():
                result_queue.put((client_id, stream_name, frame_id, outputs[0], 
                                  original_shape, scale, x_offset, y_offset, receive_time), 
                                  timeout=1.0)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Inference thread: {e}")

# --- 3. Post-processing Thread ---
def post_processing_worker(result_queue, client_results, stop_event):
    """
    Worker thread that performs post-processing (NMS) and formats the output.
    """
    while not stop_event.is_set():
        try:
            # Get data from inference
            (client_id, stream_name, frame_id, outputs, 
             original_shape, scale, x_offset, y_offset, receive_time) = result_queue.get(timeout=1.0)
            
            with Timer(f"{stream_name} (Frame {frame_id}) - Post-processing"):
                preds = np.transpose(outputs[0], (0, 2, 1))[0] # (1, 84, 8400) -> (8400, 84)
                
                boxes = []
                confidences = []
                class_ids = []

                for det in preds:
                    class_scores = det[4:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]

                    if confidence > CONF_THRESHOLD:
                        cx, cy, w, h = det[:4]
                        x1 = int(cx - w / 2)
                        y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)
                        
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(int(class_id))
                
                indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

                # --- Build formatted detections ---
                final_detections = []
                if len(indices) > 0:
                    for i in indices.flatten():
                        x1, y1, x2, y2 = boxes[i]
                        
                        # Rescale boxes back to original image size
                        x1_orig = (x1 - x_offset) / scale
                        y1_orig = (y1 - y_offset) / scale
                        x2_orig = (x2 - x_offset) / scale
                        y2_orig = (y2 - y_offset) / scale
                        
                        # Clip to original image boundaries
                        orig_h, orig_w = original_shape
                        x1_final = round(max(0, min(x1_orig, orig_w)), 2)
                        y1_final = round(max(0, min(y1_orig, orig_h)), 2)
                        x2_final = round(max(0, min(x2_orig, orig_w)), 2)
                        y2_final = round(max(0, min(y2_orig, orig_h)), 2)
                        
                        class_name = CLASSES[class_ids[i]]
                        conf = round(confidences[i], 2)
                        
                        detection = {
                            "label": class_name,
                            "conf": conf,
                            "bbox": [x1_final, y1_final, x2_final, y2_final]
                        }
                        final_detections.append(detection)
            
            # --- Calculate final latency ---
            total_latency_ms = round((time.perf_counter() - receive_time) * 1000, 1)

            # --- Build the final JSON payload ---
            result_payload = {
                "timestamp": int(time.time()),
                "frame_id": frame_id,
                "stream_name": stream_name,
                "latency_ms": total_latency_ms,
                "detections": final_detections
            }
            
            # Store final result for this client
            client_results[client_id] = result_payload

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Post-processing thread: {e}")


# --- 4. WebSocket Handler ---
# clients format: {websocket: (client_id, frame_count)}
clients = {} 
# client_results format: {client_id: result_payload}
client_results = {} 
# raw_frame_queue format: (client_id, stream_name, frame_id, raw_data, receive_time)
raw_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE * 2)
next_client_id = 0

async def handler(websocket, path):
    global next_client_id
    
    client_id = next_client_id
    next_client_id += 1
    stream_name = f"cam_{client_id}"
    frame_count = 0
    
    clients[websocket] = (client_id, frame_count)
    client_results[client_id] = {} # Init empty result
    print(f"[INFO] Client {client_id} ({stream_name}) connected from {websocket.remote_address}")

    try:
        while True:
            # --- Receive Frame ---
            try:
                raw_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                receive_time = time.perf_counter()
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            
            # Update client's frame count
            frame_count += 1
            clients[websocket] = (client_id, frame_count)

            # --- Push to processing pipeline ---
            if not raw_frame_queue.full():
                raw_frame_queue.put_nowait((client_id, stream_name, frame_count, raw_data, receive_time))
            else:
                print(f"[WARN] {stream_name}: RAW frame queue is full. Server is overloaded. Dropping frame {frame_count}.")

            # --- Send Result (non-blocking) ---
            result = client_results.get(client_id)
            if result:
                await websocket.send(json.dumps(result))
    
    except Exception as e:
        print(f"[ERROR] Handler: {e}")
    finally:
        print(f"[INFO] Client {client_id} ({stream_name}) disconnected")
        del clients[websocket]
        del client_results[client_id]


# --- Main Application ---
async def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model file not found: {MODEL_PATH}")
        print("[FATAL] Please run export_model.py first!")
        return

    stop_event = threading.Event()
    
    # Queues
    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    # Start worker threads
    all_threads = []
    
    for _ in range(PROCESSING_THREADS):
        t = threading.Thread(target=frame_receiver, args=(frame_queue, raw_frame_queue, stop_event), daemon=True)
        t.start()
        all_threads.append(t)
        
    for _ in range(INFERENCE_THREADS):
        t = threading.Thread(target=inference_worker, args=(frame_queue, result_queue, stop_event, MODEL_PATH), daemon=True)
        t.start()
        all_threads.append(t)
        
    for _ in range(PROCESSING_THREADS):
        t = threading.Thread(target=post_processing_worker, args=(result_queue, client_results, stop_event), daemon=True)
        t.start()
        all_threads.append(t)

    print("[INFO] All worker threads started.")

    # Start WebSocket server
    async with websockets.serve(handler, "localhost", 8765, max_size=1_000_000): # 1MB max msg size
        print("[INFO] Server started at ws://localhost:8765")
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            stop_event.set()
            for t in all_threads:
                t.join()
            print("[INFO] Server shut down gracefully.")

if __name__ == "__main__":
    asyncio.run(main())