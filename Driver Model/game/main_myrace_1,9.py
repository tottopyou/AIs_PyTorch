import pygame
import sys
import math
import time
from shader import *
from pygame.locals import *
import socket
import select
import pickle

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server's IP address and port
SERVER_HOST = '127.0.0.1'  # localhost
SERVER_PORT = 65432        # Port to connect to

try:
    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))
except ConnectionRefusedError:
    print("Connection refused. Make sure the server is running.")
    sys.exit()

pygame.init()

bg = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0,255,0)

ww = pygame.display.Info().current_w
wh = pygame.display.Info().current_h

window = pygame.display.set_mode((ww, wh))
pygame.display.set_caption("MyRace - [Space] to switch day/night")
window.fill(bg)

shade = shader(window)
shade.setup((7, 7, 7))

stretch = []
a = 1
track_v = 11

while a == 1:
    try:
        exec("stretch.append(pygame.image.load('tracks/track_" + str(track_v) + ".png'))")
    except:
        a = 0
    track_v += 1

c_fence = (255, 5, 5, 255)
c_finish = (255, 255, 5, 255)
c_reward = (0, 255, 0, 255)

counter = 0

player_1 = pygame.Rect(100, 165, 20, 20)
image_1 = pygame.image.load("car_1.png")

explosion = pygame.image.load("explosion.png")

pressed_1 = False
pressed_1_l = False
pressed_1_r = False
pressed_1_b = False

counter_1 = 0
angle_1 = 0
destroy_1 = 0
count_destr_1 = 0

mvsp = 10
angle_ch = 4 / 8  # change angle

acceleration = 0
brake = 0
lose = 0
win = 0
reward = 0

# Define raycasting parameters
RAY_COUNT = 12
RAY_ANGLE_OFFSET = 30  # Initial angle offset
RAY_LENGTH = 200  # Maximum length of ray
ray_distances = [RAY_LENGTH] * RAY_COUNT
finish_distance = [0] * RAY_COUNT

clock = pygame.time.Clock()
fps = 120
time_ = 0

x = 1
while x == 1:
    if count_destr_1 == 1:
        player_1.left = 100
        player_1.top = 165

    # Player 1
    if count_destr_1 == 0:
        if pressed_1 and counter_1 < mvsp:
            counter_1 += 0.25
            acceleration = 1
        else:
            acceleration = 0

        if pressed_1_b:
            counter_1 -= 0.25
            brake = 1
        else:
            brake = 0

        if pressed_1_l and counter_1 > 2:
            angle_1 -= angle_ch * counter_1
        elif pressed_1_l and counter_1 < -2:
            angle_1 -= angle_ch * counter_1

        if pressed_1_r and counter_1 > 2:
            angle_1 += angle_ch * counter_1
        elif pressed_1_r and counter_1 < -2:
            angle_1 += angle_ch * counter_1

        if pressed_1 == False and counter_1 > 0:
            counter_1 -= 0.25
        if pressed_1_b == False and counter_1 < 0:
            counter_1 += 0.25

        b_1 = math.cos(math.radians(angle_1)) * counter_1
        a_1 = math.sin(math.radians(angle_1)) * counter_1
        player_1.left += round(b_1)
        player_1.top += round(a_1)

        image_1_neu = pygame.transform.rotate(image_1, angle_1 * -1)

        # Raycasting
        for i in range(RAY_COUNT):
            angle = math.radians(angle_1 + RAY_ANGLE_OFFSET * i)
            x1, y1 = player_1.center
            ray_distances[i] = RAY_LENGTH
            finish_distance[i] = 0

            for j in range(RAY_LENGTH):
                end_x = int(x1 + j * math.cos(angle))
                end_y = int(y1 + j * math.sin(angle))

                if end_x < 0 or end_x >= ww or end_y < 0 or end_y >= wh:
                    break

                if window.get_at((end_x, end_y)) == c_fence:
                    ray_distances[i] = j
                    break

                if window.get_at((end_x, end_y)) == c_finish:
                    ray_distances[i] = j
                    finish_distance[i] = j
                    break
    else:
        count_destr_1 -= 1

    for event in pygame.event.get():
        if event.type == QUIT:
            x = 0

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                x = 0

            if event.key == K_RETURN:
                counter += 1
                player_1.left = 100
                player_1.top = 165
                angle_1 = 0

                if counter >= len(stretch):
                    counter = 0

            if event.key == K_UP:
                pressed_1 = True
            if event.key == K_LEFT:
                pressed_1_l = True
            if event.key == K_RIGHT:
                pressed_1_r = True
            if event.key == K_DOWN:
                pressed_1_b = True

        if event.type == KEYUP:
            if event.key == K_UP:
                pressed_1 = False
            if event.key == K_LEFT:
                pressed_1_l = False
            if event.key == K_RIGHT:
                pressed_1_r = False
            if event.key == K_DOWN:
                pressed_1_b = False

    #ray_info = ", ".join([f"Ray {i + 1}: {round(dist, 2)}" for i, dist in enumerate(ray_distances)])
    #data_to_send = f"Speed: {round(counter_1, 2)}, Acceleration: {acceleration}, Brake: {brake}, " \
    #               f"Angle: {round(angle_1, 2)}, Lose: {lose}, Win: {win}, Reward: {reward}, {ray_info} "


    readable, _, _ = select.select([client_socket], [], [], 0)
    if readable:
        data = client_socket.recv(1)
        if data:
            action = int(data.decode())
            #print("Received action from server:", action)

            if action == 0:  # Backward
                pressed_1 = False
                pressed_1_b = True
                pressed_1_l = False
                pressed_1_r = False
            elif action == 1 :  # Forward
                pressed_1 = True
                pressed_1_b = False
                pressed_1_l = False
                pressed_1_r = False
            elif action == 2:  # Left
                pressed_1 = False
                pressed_1_b = False
                pressed_1_l = True
                pressed_1_r = False
            elif action == 3:  # Right
                pressed_1 = False
                pressed_1_b = False
                pressed_1_l = False
                pressed_1_r = True

    window.fill((0, 0, 0))
    window.blit(stretch[counter], (0, 0))
    if count_destr_1 == 0:
        try:

            if window.get_at((player_1.left + 10, player_1.top + 10)) == c_fence:
                destroy_1 = 1
                lose = 1

            if window.get_at((player_1.left + 10, player_1.top + 10)) == c_finish:
                destroy_1 = 1
                win = 1

            if window.get_at((player_1.left + 10, player_1.top + 10)) == c_reward:
                reward = 1
            else:
                reward = 0

        except:
            destroy_1 = 1

        if destroy_1 == 0:
            window.blit(image_1_neu, player_1)


    ray_info =[round(dist, 2) for dist in ray_distances]
    ray_info_finish = [round(dist, 2) for dist in finish_distance]
    data_to_send = [round(counter_1, 2), acceleration, brake, round(angle_1, 2), lose, win, reward] + ray_info + ray_info_finish

    #print(data_to_send)
    data_bytes = pickle.dumps(data_to_send)
    client_socket.sendall(data_bytes)

    font = pygame.font.Font(None, 36)
    text_speed = font.render(f"Speed: {round(counter_1, 2)}", True, blue)
    text_acceleration = font.render(f"Acceleration: {acceleration}", True, blue)
    text_brake = font.render(f"Brake: {brake}", True, blue)
    text_angle = font.render(f"Angle: {round(angle_1, 2)}", True, blue)
    text_lose = font.render(f"Lose: {round(lose, 2)}", True, blue)
    text_reward = font.render(f"Reward: {round(reward, 2)}", True, blue)

    window.blit(text_speed, (10, 10))
    window.blit(text_acceleration, (10, 50))
    window.blit(text_brake, (10, 90))
    window.blit(text_angle, (10, 130))
    window.blit(text_lose, (10, 170))
    window.blit(text_reward, (10, 210))


    # Display ray distances and angles
    for i in range(RAY_COUNT):
        angle = math.radians(angle_1 + RAY_ANGLE_OFFSET * i)
        end_x = player_1.centerx + ray_distances[i] * math.cos(angle)
        end_y = player_1.centery + ray_distances[i] * math.sin(angle)

        if finish_distance[i] > 0 :
            draw_color = green
        else:
            draw_color = blue

        pygame.draw.line(window, draw_color, player_1.center, (end_x, end_y), 2)

        # Display ray info
        ray_info = font.render(f"Ray {i + 1}: Angle = {round(angle_1 + RAY_ANGLE_OFFSET * i, 2)}, "
                               f"Wall Distance = {ray_distances[i]}, Finish Distance = {finish_distance[i]}", True,
                               draw_color)
        window.blit(ray_info, (10, 250 + 30 * i))

    if destroy_1 == 1:
        time.sleep(0.1)
        window.blit(explosion, player_1)
        pygame.display.update()
        destroy_1 = 0
        angle_1 = 0
        lose = 0
        win = 0
        count_destr_1 = 25

    pygame.display.update()
    clock.tick(fps)

pygame.quit()
