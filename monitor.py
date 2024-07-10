# monitor.py
# handles monitoring and logging of CPU/GPU and memory usage

import psutil
import GPUtil
import time
from flask import Flask, jsonify
import threading

app = Flask(__name__)

# Global variables to store monitoring data
cpu_usage = []
memory_usage = []
gpu_usage = []
gpu_memory = []
monitoring_thread = None
stop_monitoring_flag = False

def monitor_system(interval=1.0):
    global cpu_usage, memory_usage, gpu_usage, gpu_memory, stop_monitoring_flag
    while not stop_monitoring_flag:
        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage.append(gpus[0].load * 100)
            gpu_memory.append(gpus[0].memoryUtil * 100)

        time.sleep(interval)

@app.route('/start_monitoring', methods=['GET'])
def start_monitoring():
    global monitoring_thread, stop_monitoring_flag
    if monitoring_thread is None or not monitoring_thread.is_alive():
        stop_monitoring_flag = False
        monitoring_thread = threading.Thread(target=monitor_system)
        monitoring_thread.start()
    return jsonify({"message": "Monitoring started"}), 200

@app.route('/stop_monitoring', methods=['GET'])
def stop_monitoring():
    global stop_monitoring_flag
    stop_monitoring_flag = True
    return jsonify({"message": "Monitoring stopped"}), 200

@app.route('/data', methods=['GET'])
def get_data():
    global cpu_usage, memory_usage, gpu_usage, gpu_memory
    return jsonify({
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_usage": gpu_usage,
        "gpu_memory": gpu_memory
    })

if __name__ == '__main__':
    app.run(port=5000)
