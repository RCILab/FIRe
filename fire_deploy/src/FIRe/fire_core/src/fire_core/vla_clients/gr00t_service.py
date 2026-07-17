import threading
from queue import Queue
from typing import Any, Dict

import numpy as np
import zmq
from .gr00t.base import BaseInferenceClient, TorchSerializer


# The GR00T server returns the gripper (action.gripper_close) using the [0, 1]
# convention (0=close, 1=open), but FIRe/the controller expects the [-1, +1]
# convention (-1=close, +1=open). The C++ VLA action server's close check is
# `filtered_gripper < 0.0`, so if the server sends close as 0.0 the condition is
# never met and the gripper never closes. The linear remap 2*g-1 aligns the conventions (0 -> -1 close, 1 -> +1 open).
_GRIPPER_KEY = "action.gripper_close"


def _remap_gripper_convention(response: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(response, dict) and _GRIPPER_KEY in response:
        g = np.asarray(response[_GRIPPER_KEY], dtype=np.float32)
        response[_GRIPPER_KEY] = g * 2.0 - 1.0
    return response


class AsyncGr00tInferenceClient(BaseInferenceClient):
    """
    A client that implements asynchronous communication by inheriting from BaseInferenceClient.

    It uses a separate worker thread to request actions from the server and receive results.
    This allows the main thread (Isaac Lab) to continue the simulation
    without waiting for network responses.
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        # Do not call the BaseInferenceClient's __init__ directly as it creates a socket.
        # Instead, initialize only the necessary attributes.
        self.context = zmq.Context.instance()  # Use a global instance for thread safety
        self.host = host
        self.port = port
        self.api_token = api_token
        self.socket_address = f"tcp://{self.host}:{self.port}"

        # Queues for inter-thread data exchange (maxsize=1 to ensure pipeline)
        self.request_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        
        # Create and start the worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"Async worker thread started. Connecting to {self.socket_address}...")

    def _worker_loop(self):
        """The main loop of the worker thread that runs in the background and communicates with the server."""
        # A socket must be created and used within the thread.
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.socket_address)
        
        while True:
            # 1. Get observations from the queue sent by the Main Thread (waits if empty)
            observations = self.request_queue.get()
            if observations is None:  # Termination signal
                break

            try:
                # 2. Perform the logic of BaseInferenceClient's call_endpoint here
                request = {"endpoint": "get_action", "data": observations}
                if self.api_token:
                    request["api_token"] = self.api_token

                socket.send(TorchSerializer.to_bytes(request))
                
                # 3. Receive the response (the worker thread blocks here)
                message = socket.recv()
                response = TorchSerializer.from_bytes(message)

                if "error" in response:
                    raise RuntimeError(f"Server error: {response['error']}")
                
                # 4. Put the result in the result queue
                self.result_queue.put(response)

            except Exception as e:
                print(f"[Worker Thread Error] {e}")
                self.result_queue.put({"error": str(e)})  # Also pass the error

        socket.close()
        print("Async worker thread stopped.")

    def request_action(self, observations: Dict[str, Any]):
        """
        [Non-blocking] Proactively requests the next action.
        Puts the observation into the request queue and returns immediately.
        """
        self.request_queue.put(observations)
        self.result_queue.queue.clear()

    def get_result(self) -> Dict[str, Any]:
        """
        [Blocking] Gets the result of the proactively requested action.
        Waits until a result is available in the queue and then returns it.
        """
        result = self.result_queue.get()
        if "error" in result:
             raise RuntimeError(f"Received error from server or worker: {result['error']}")
        return _remap_gripper_convention(result)

    def close(self):
        """Called when the client is shut down to clean up the worker thread."""
        if self.worker_thread.is_alive():
            self.request_queue.put(None)  # Send termination signal
            self.worker_thread.join(timeout=2)  # Wait for the thread to terminate for up to 2 seconds
        self.context.term()

    def __del__(self):
        """Safely cleans up resources when the object is destroyed."""
        self.close()

    def get_action_sync(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Synchronous] Creates a socket for a one-time request and waits for the result.
        Use only when an immediate result is needed, such as during a reset().
        """
        # Create a temporary socket to be used only within this method
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.socket_address)
        
        result = {}
        try:
            # Directly implement the call_endpoint logic here
            request = {"endpoint": "get_action", "data": observations}
            if self.api_token:
                request["api_token"] = self.api_token

            socket.send(TorchSerializer.to_bytes(request))
            message = socket.recv()
            response = TorchSerializer.from_bytes(message)

            if "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            
            result = response
        
        finally:
            # Ensure the socket is always closed.
            socket.close()

        return _remap_gripper_convention(result)
