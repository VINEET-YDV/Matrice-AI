import cv2
import asyncio
import websockets
import time
import threading
import queue
import json
import argparse
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Global Queue ---
# This queue decouples the frame grabber (blocking I/O)
# from the network sender (async I/O).
# maxsize=1 ensures we *always* send the latest frame.
frame_queue = queue.Queue(maxsize=1)

# --- Performance Metrics ---
stats = {
    "frames_captured": 0,
    "frames_sent": 0,
    "frames_received": 0,
    "total_latency_ms": 0
}
stats_lock = threading.Lock()

# --- Frame Grabber Thread ---
def frame_grabber_thread(source, stop_event):
    """
    A dedicated thread to grab frames from the video source.
    This is blocking I/O, so it runs in a thread to not
    block the main asyncio event loop.
    """
    global frame_queue, stats, stats_lock
    threading.current_thread().name = "FrameGrabber"
    
    try:
        source = int(source)
    except ValueError:
        pass # Keep as string for file/RTSP
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"Failed to open video source: {source}")
        stop_event.set()
        return
    
    logging.info(f"Video source opened: {source}")
    
    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream or disconnected. Stopping.")
                stop_event.set()
                break

            # Non-blocking put:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                # This is expected. It just means the network
                # is slower than the camera, so we drop a frame.
                pass
            
            with stats_lock:
                stats["frames_captured"] += 1
            
            # Limit capture FPS to ~60 to not needlessly burn CPU
            time.sleep(1/60.0) 

        except Exception as e:
            logging.error(f"Error in FrameGrabber: {e}")
            stop_event.set()
    
    cap.release()
    logging.info("Frame grabber thread stopped.")

# --- AsyncIO Tasks ---

async def sender_task(websocket, stop_event):
    """
    Coroutine: Gets frames from the queue, compresses, and sends them.
    """
    global frame_queue, stats, stats_lock
    loop = asyncio.get_running_loop()

    while not stop_event.is_set():
        try:
            # Use run_in_executor for the blocking queue.get()
            frame = await loop.run_in_executor(None, frame_queue.get, True, 0.1)
            
            # 1. Compress frame to JPEG
            # 80% quality is a good trade-off
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                logging.warning("Failed to encode frame to JPEG.")
                continue

            # 2. Prepend 8-byte timestamp (nanoseconds)
            timestamp_ns = time.time_ns()
            payload = timestamp_ns.to_bytes(8, 'big') + buffer.tobytes()

            # 3. Send over WebSocket
            await websocket.send(payload)
            
            with stats_lock:
                stats["frames_sent"] += 1

        except queue.Empty:
            # No frame available, just wait
            await asyncio.sleep(0.001)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("Sender: Connection closed.")
            stop_event.set()
            break
        except Exception as e:
            logging.error(f"Error in sender_task: {e}")
            stop_event.set()

async def receiver_task(websocket, output_file, stop_event):
    """
    Coroutine: Listens for JSON results, calculates latency, and saves to file.
    """
    global stats, stats_lock
    
    with open(output_file, 'w') as f:
        f.write("[\n") # Start a JSON array
        first_result = True

        while not stop_event.is_set():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                
                # 1. Calculate round-trip latency
                reception_time_ns = time.time_ns()
                json_result = json.loads(message)
                send_time_ns = json_result.get("timestamp_ns")
                
                latency_ms = (reception_time_ns - send_time_ns) / 1_000_000
                
                with stats_lock:
                    stats["frames_received"] += 1
                    stats["total_latency_ms"] += latency_ms
                
                # 2. Save result to JSON file
                if not first_result:
                    f.write(",\n")
                json.dump(json_result, f, indent=2)
                first_result = False

            except asyncio.TimeoutError:
                # No message received, just check if we should stop
                continue
            except websockets.exceptions.ConnectionClosed:
                logging.warning("Receiver: Connection closed.")
                stop_event.set()
                break
            except Exception as e:
                logging.error(f"Error in receiver_task: {e}")
                stop_event.set()
        
        f.write("\n]\n") # Close the JSON array

async def metrics_task(stop_event):
    """Coroutine: Prints performance metrics every 2 seconds."""
    global stats, stats_lock
    last_time = time.time()
    last_stats = stats.copy()

    while not stop_event.is_set():
        await asyncio.sleep(2.0)
        
        current_time = time.time()
        delta_time = current_time - last_time
        
        with stats_lock:
            current_stats = stats.copy()
        
        delta_captured = current_stats["frames_captured"] - last_stats["frames_captured"]
        delta_sent = current_stats["frames_sent"] - last_stats["frames_sent"]
        delta_received = current_stats["frames_received"] - last_stats["frames_received"]
        delta_latency = current_stats["total_latency_ms"] - last_stats["total_latency_ms"]
        
        capture_fps = delta_captured / delta_time
        send_fps = delta_sent / delta_time
        receive_fps = delta_received / delta_time
        
        avg_latency = (delta_latency / delta_received) if delta_received > 0 else 0
        
        logging.info(
            f"Metrics: Capture={capture_fps:.1f} FPS | "
            f"Send={send_fps:.1f} FPS | "
            f"Receive={receive_fps:.1f} FPS | "
            f"Avg. Latency={avg_latency:.2f} ms"
        )
        
        last_time = current_time
        last_stats = current_stats

async def run_client(source, server_url, output_file):
    """Main async function to coordinate all tasks."""
    stop_event = threading.Event()

    # Start the blocking frame grabber in its own thread
    grabber = threading.Thread(
        target=frame_grabber_thread, 
        args=(source, stop_event), 
        daemon=True
    )
    grabber.start()

    while not stop_event.is_set():
        try:
            async with websockets.connect(server_url, ping_interval=5, ping_timeout=20) as websocket:
                logging.info(f"Connected to server: {server_url}")
                
                # Start the async tasks
                sender = asyncio.create_task(sender_task(websocket, stop_event))
                receiver = asyncio.create_task(receiver_task(websocket, output_file, stop_event))
                metrics = asyncio.create_task(metrics_task(stop_event))
                
                await asyncio.gather(sender, receiver, metrics)
                
        except (websockets.exceptions.ConnectionClosedError, OSError) as e:
            logging.warning(f"Connection lost: {e}. Retrying in 3 seconds...")
            stop_event.clear() # Clear stop if it was set by connection loss
            await asyncio.sleep(3)
        except Exception as e:
            logging.error(f"Unhandled client error: {e}")
            stop_event.set()
            
    logging.info("Client shutting down.")
    grabber.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Client")
    parser.add_argument("--source", required=True, help="Video source (webcam ID, file path, or RTSP URL)")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--output", default="results.json", help="Output JSON file for results")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_client(args.source, args.server, args.output))
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")