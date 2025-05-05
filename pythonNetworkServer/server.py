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
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
                print(f"[Server] Active connections: {threading.active_count() - 1}")
        except KeyboardInterrupt:
            print("\n[Server] Shutting down due to Ctrl+C...")
            self.server_socket.close()
        except Exception as e:
            print(f"[Server] Error: {e}")
            self.server_socket.close()


    def handle_client(self,conn, addr):
        print(f"[+] Connected by {addr}")
        model = None
        with conn, conn.makefile('r') as client_file:
            for line in client_file:
                try:
                    msg = json.loads(line.strip())
                    msgType = MessageType(msg["type"])
                    if msgType == MessageType.SET_WEIGHTS:
                        self.handle_set_weights_message(model,msg["msg"])
                    elif msgType == MessageType.GET_WEIGHTS:
                        self.handle_get_weights_message(model,conn)
                    elif msgType == MessageType.SEND_DATA:
                        model = self.handle_set_data_message(msg["msg"])
                    else:
                        print(msg)
                except json.JSONDecodeError as e:
                    print(f"[{addr}] Failed to decode JSON:", e)

    def handle_set_weights_message(self,model,msg):
        if not model:
            return
        model.set_weights(msg["weights"]) 
    def handle_get_weights_message(self,model,conn):
        if not model:
            return
        weights = model.get_weights()
        
        message = json.dumps({"weights": weights})  # '\n' helps with framing
        conn.sendall(message.encode('utf-8'))

    def handle_set_data_message(self,msg):
        return network.GRU(msg["X"],msg["Y"])


if __name__ == "__main__":
    server = Server()
    server.start_server()