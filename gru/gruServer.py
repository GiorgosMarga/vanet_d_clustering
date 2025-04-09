import socket
import subprocess
import os 
import re
import numpy as np
import pickle
import json

def update_weights(weights):
    with os.scandir("weights") as entries:
        for i, entry in enumerate(entries):
            if entry.is_file():
                with open(entry.path, "w") as f:
                    np.savetxt(entry.path, weights[i][0])


def Train(weights, epochs, first_train):

    if first_train != "start":
        update_weights(weights)

    subprocess.run(["python", "gru.py", first_train, str(epochs)])

    weight_matrixes = []

    with os.scandir("weights") as entries:
        for entry in entries:
            if entry.is_file():
                with open(entry.path, 'r') as f:
                    content = f.read()
                    content = re.split(r'\s+', content)
                    content = content[:-1] # ignore last line always empty
                    weight_matrixes.append(content)
   
    return weight_matrixes


# Server settings
HOST = '192.168.1.3'  # or '0.0.0.0' to listen on all interfaces
PORT = 4004

# Create UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

first_train = "start"

try:
    while True:
        print(f"UDP server listening on {HOST}:{PORT}")
        # Wait for incoming message
        data, addr = server_socket.recvfrom(65535)  # Buffer size
        print("RECIEVED\n")
        received_list_of_arrays = json.loads(data.decode())
        # received_list_of_arrays = data.decode()
        print(f"Received from {addr}")

        #train with gru
        new_weights = Train(received_list_of_arrays, 1, first_train)
        first_train = "False"
        
        # Send back weights
        response = json.dumps(new_weights)
        server_socket.sendto(response.encode(), addr)

except KeyboardInterrupt:
    print("\nShutting down UDP server.")
finally:
    server_socket.close()

