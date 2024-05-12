import socket

# Define the server's IP address and port
HOST = '127.0.0.1'  # localhost
PORT = 65432  # Port to listen on


# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the address and port
    server_socket.bind((HOST, PORT))

    # Listen for incoming connections
    server_socket.listen()

    print("Server listening on", HOST, PORT)

    # Accept a connection
    conn, addr = server_socket.accept()
    with conn:
        print('Connected by', addr)

        while True:
            # Receive data from the client
            data = conn.recv(1024)
            if not data:
                break

            # Print the received data
            print("Received data:", data.decode())
