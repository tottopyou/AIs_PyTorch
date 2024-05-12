import socket
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
import time

# Define the server's IP address and port
HOST = '127.0.0.1'  # localhost
PORT = 65432  # Port to listen on

MAX_ACTIONS_PER_SECOND = 60
MIN_ACTION_INTERVAL = 1 / MAX_ACTIONS_PER_SECOND

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

        data = conn.recv(1024)
        print("Received data:", data.decode())
        class DQN(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # Function to process data and create tensor
        def process_data(data):
            # Define regex patterns
            speed_pattern = r"Speed: ([\d.]+)"
            acceleration_pattern = r"Acceleration: ([\d.]+)"
            brake_pattern = r"Brake: ([\d.]+)"
            angle_pattern = r"Angle: ([\d.]+)"
            lose_pattern = r"Lose: ([\d.]+)"
            win_pattern = r"Win: ([\d.]+)"
            reward_pattern = r"Reward: ([\d.]+)"
            ray_pattern = r"Ray (\d+): ([\d.]+)"

            # Extract values using regex
            speed = float(re.search(speed_pattern, data).group(1))
            acceleration = float(re.search(acceleration_pattern, data).group(1))
            brake = float(re.search(brake_pattern, data).group(1))
            angle = float(re.search(angle_pattern, data).group(1))
            lose = float(re.search(lose_pattern, data).group(1))
            win = float(re.search(win_pattern, data).group(1))
            reward_line = float(re.search(reward_pattern, data).group(1))
            rays = [float(match.group(2)) for match in re.finditer(ray_pattern, data)]

            tensor_data = torch.tensor([speed, acceleration, brake, angle, lose, win, reward_line] + rays, dtype=torch.float32)
            return tensor_data

        learning_rate = 0.001
        gamma = 0.99  # Discount factor
        epsilon = 0.1  # Epsilon-greedy exploration rate
        batch_size = 32
        epoches = 1000

        input_size = 15  # 7 original features + 8 rays
        output_size = 4  # 4 actions (forward, backward, left, right)


        model = DQN(input_size, output_size)
        target_model = DQN(input_size, output_size)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        memory =[]

        try:
            model.load_state_dict(torch.load("dqn_model.pth"))
            target_model.load_state_dict(model.state_dict())
            print("Loaded previously saved model weights")
        except FileNotFoundError:
            print("No saved model weights found")

        for episode in range(epoches):

            state = process_data(data.decode())

            episode_reward = 0
            episode_loss = 0

            last_action_time = time.time()

            win = state[5].item()
            lose = state[4].item()
            reward_line = state[6].item()

            for step in range(output_size):
                if random.random() < epsilon:
                    action = random.randint(0, output_size - 1)
                else:
                    with torch.no_grad():
                        q_values = model(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()

                print(action)
                conn.sendall(str(action).encode())

                time_passed = time.time() - last_action_time

                remaining_time = MIN_ACTION_INTERVAL - time_passed

                if remaining_time > 0:
                    time.sleep(remaining_time)

                last_action_time = time.time()

                observation = process_data(data.decode())

                if win == 1:
                    reward = 100
                elif reward_line == 1:
                    reward = 10
                elif lose == 1:
                    reward = -10
                else:
                    reward = -0.1

                memory.append((state, action, reward, observation))

                episode_reward += reward

                state = observation

                # Sample a batch of experiences from memory
                if len(memory) < batch_size:
                    continue
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = zip(*batch)

                # Convert to tensors
                states = torch.stack(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.stack(next_states)

                q_values = model(states)
                next_q_values = target_model(next_states)
                q_values_next = rewards + gamma * torch.max(next_q_values, dim=1)[0]

                q_values_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = criterion(q_values_action, q_values_next)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 60 == 0:
                    target_model.load_state_dict(model.state_dict())

                episode_loss += loss.item()

            print(
                f"Episode {episode + 1}: Total Reward = {episode_reward}, Average Loss = {episode_loss / output_size}")

            # Save trained model weights
            torch.save(model.state_dict(), "dqn_model.pth")