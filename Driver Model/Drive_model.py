import socket
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pickle

# Define the server's IP address and port
HOST = '127.0.0.1'  # localhost
PORT = 65432  # Port to listen on

MAX_ACTIONS_PER_SECOND = 90
MIN_ACTION_INTERVAL = 1 / MAX_ACTIONS_PER_SECOND

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Active device: ", device)

# Define the neural network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.view(-1, 31)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to process data and create tensor
def process_data(data):
    # Extract values using regex
    speed = float(data[0]) / 10.0
    acceleration = float(data[1])
    brake = float(data[2])
    angle = float(data[3]) / 360.0
    lose = float(data[4])
    win = float(data[5])
    reward_line = float(data[6])
    rays = [float(data[i] / 200.0) for i in range(7, len(data))]

    tensor_data = torch.tensor([speed, acceleration, brake, angle, lose, win, reward_line] + rays,
                               dtype=torch.float32)
    return tensor_data

learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.1  # Epsilon-greedy exploration rate

input_size = 31  # 7 original features + 8 rays
output_size = 4 # 4 actions (forward, backward, left, right)
epoches = 1000000

model = DQN(input_size, output_size).to(device)
target_model = DQN(input_size, output_size).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

try:
    model.load_state_dict(torch.load("../dqn_model.pth"))
    target_model.load_state_dict(model.state_dict())
    print("Loaded previously saved model weights")
except FileNotFoundError:
    print("No saved model weights found")

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

        last_action_time = time.time()

        for episode in range(epoches):

            received_data = conn.recv(4096)
            data = pickle.loads(received_data)

            state = process_data(data).to(device)

            episode_reward = 0
            episode_loss = 0

            win = state[5].item()
            lose = state[4].item()
            reward_line = state[6].item()
            speed = state[0].item()

            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))
                    action = torch.argmax(q_values).item()

            conn.sendall(str(action).encode())

            time_passed = time.time() - last_action_time

            remaining_time = MIN_ACTION_INTERVAL - time_passed

            if remaining_time > 0:
                time.sleep(remaining_time)

            last_action_time = time.time()

            observation = process_data(data).to(device)\

            reward = 0

            if win == 1:
                print("Winner")
                reward += 2
            if lose == 1:
                print("You are dead")
                reward += -0.6
            if reward_line == 1 and speed > 0.5:
                print("Reward yepi")
                reward += 0.4
            if speed < 2:
                reward += -0.05
            if speed > 4:
                reward += 0.1

            episode_reward += reward

            state = observation

            # Train the model with the most recent transition
            state = state.unsqueeze(0).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward).unsqueeze(0).to(device)

            q_values = model(state)
            next_q_values = target_model(state)
            q_values_next = reward + gamma * torch.max(next_q_values, dim=1)[0]

            q_values_action = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            loss = criterion(q_values_action, q_values_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if episode % 60 == 0:
                target_model.load_state_dict(model.state_dict())

            episode_loss += loss.item()

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Average Loss = {episode_loss}")

            # Save trained model weights
            if (episode + 1) % 100 == 0:
                torch.save(model.state_dict(), "dqn_model.pth")
