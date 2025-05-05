import socket
import json
import threading
import network
from enum import Enum
HOST = '127.0.0.1'  # Localhost
PORT = 5000       # Arbitrary non-privileged port

class MessageType(Enum):
    SET_WEIGHTS = 0
    GET_WEIGHTS = 1
    TRAIN = 2
    EVALUATE = 3
    PREDICT = 4
    SEND_DATA = 5

class GoClient():
    def __init__(self,conn: socket,address: str):
        self.conn = conn
        self.model = None
        self.address = address
    
    def handle_conn(self):
        print(f"[+] Connected by {self.address}")
        with self.conn, self.conn.makefile('r') as client_file:
            for line in client_file:
                try:
                    msg = json.loads(line.strip())
                    msgType = MessageType(msg["type"])
                    if msgType == MessageType.SET_WEIGHTS:
                        self._handle_set_weights_message(msg["msg"])
                    elif msgType == MessageType.GET_WEIGHTS:
                        self._handle_get_weights_message()
                    elif msgType == MessageType.SEND_DATA:
                        self._handle_set_data_message(msg["msg"])
                    elif msgType == MessageType.TRAIN:
                        self._handle_train_message(msg["msg"])
                    elif msgType == MessageType.EVALUATE:
                        self._handle_evaluate_message()
                    elif msgType == MessageType.PREDICT:
                        self._handle_predict_message(msg["msg"])
                    else:
                        print(msg)
                except json.JSONDecodeError as e:
                    print(f"[{self.address}] Failed to decode JSON:", e)

    def _handle_evaluate_message(self):
        if not self.model:
            return
        return self.model.evaluate()
        
    def _handle_set_weights_message(self,msg):
        if not self.model:
            return
        self.model.set_weights(msg["weights"]) 
    def _handle_get_weights_message(self):
        if not self.model:
            return
        weights = self.model.get_weights()
        
        message = json.dumps({"weights": weights})  # '\n' helps with framing
        self.conn.sendall(message.encode('utf-8'))

    def _handle_set_data_message(self,msg):
        self.model = network.GRU(msg["X"],msg["Y"])
    
    def _handle_train_message(self,msg):
        if not self.model:
            return
        epochs = msg["epochs"] if msg["epochs"] else 50
        batch_size = msg["batchSize"] if msg["batchSize"] else 10

        self.model.train(epochs,batch_size)

        self.conn.sendall("Finished Training".encode('utf-8'))

    def _handle_predict_message(self,msg):
        if not self.model:
            return

        self.model.predict(msg["x"])

        # self.conn.sendall("Finished Training".encode('utf-8'))
    

class Server():
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen()
        print(f"[Server] Listening on {HOST}:{PORT}")

    def start_server(self):
        try:
            while True:
                conn, addr = self.server_socket.accept()
                go_client = GoClient(conn=conn,address=addr)
                thread = threading.Thread(target=go_client.handle_conn())
                thread.start()
                print(f"[Server] Active connections: {threading.active_count() - 1}")
        except KeyboardInterrupt:
            print("\n[Server] Shutting down due to Ctrl+C...")
            self.server_socket.close()
            self.server_socket = None
        except Exception as e:
            print(f"[Server] Error: {e}")
            self.server_socket.close()
            self.server_socket = None
        finally:
            if self.server_socket:
                self.server_socket.close()

if __name__ == "__main__":
    server = Server()
    server.start_server()