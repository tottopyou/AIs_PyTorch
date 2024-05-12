import pygame
import sys
import math
import time
from shader import *
from pygame.locals import *
import socket
import select

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

ww = pygame.display.Info().current_w
wh = pygame.display.Info().current_h

fenster = pygame.display.set_mode((ww, wh))
pygame.display.set_caption("MyRace - [Space] to switch day/night")
fenster.fill(bg)

shade = shader(fenster)
shade.setup((7, 7, 7))

strecken = []
a = 1
a_zaehler = 9

while a == 1:
    try:
        exec("strecken.append(pygame.image.load('tracks/track_" + str(a_zaehler) + ".png'))")
    except:
        a = 0
    a_zaehler += 1

c_straße = (100, 100, 100, 255)
c_fence = (255, 5, 5, 255)
c_finish = (255, 255, 5, 255)

zaehler = 0

player_1 = pygame.Rect(100, 165, 20, 20)
image_1 = pygame.image.load("car_1.png")

explosion = pygame.image.load("explosion.png")

# Variables: Player 1
pressed_1 = False
pressed_1_l = False
pressed_1_r = False
pressed_1_b = False  # Back - Reverse

bew_zaehler_1 = 0
winkel_1 = 0
destroy_1 = 0
count_destr_1 = 0

mvsp = 10
winkel_ch = 4 / 8  # change angle

acceleration = 0
brake = 0
lose = 0
win = 0

# Define raycasting parameters
RAY_COUNT = 8
RAY_ANGLE_OFFSET = 45  # Initial angle offset
RAY_LENGTH = 200  # Maximum length of ray
ray_distances = [RAY_LENGTH] * RAY_COUNT

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
        if pressed_1 and bew_zaehler_1 < mvsp:
            bew_zaehler_1 += 0.25
            acceleration = 1
        else:
            acceleration = 0

        if pressed_1_b:
            bew_zaehler_1 -= 0.25
            brake = 1
        else:
            brake = 0

        if pressed_1_l and bew_zaehler_1 > 2:
            winkel_1 -= winkel_ch * bew_zaehler_1
        elif pressed_1_l and bew_zaehler_1 < -2:
            winkel_1 -= winkel_ch * bew_zaehler_1

        if pressed_1_r and bew_zaehler_1 > 2:
            winkel_1 += winkel_ch * bew_zaehler_1
        elif pressed_1_r and bew_zaehler_1 < -2:
            winkel_1 += winkel_ch * bew_zaehler_1

        if pressed_1 == False and bew_zaehler_1 > 0:
            bew_zaehler_1 -= 0.25
        if pressed_1_b == False and bew_zaehler_1 < 0:
            bew_zaehler_1 += 0.25

        b_1 = math.cos(math.radians(winkel_1)) * bew_zaehler_1
        a_1 = math.sin(math.radians(winkel_1)) * bew_zaehler_1
        player_1.left += round(b_1)
        player_1.top += round(a_1)

        image_1_neu = pygame.transform.rotate(image_1, winkel_1 * -1)

        # Raycasting
        for i in range(RAY_COUNT):
            angle = math.radians(winkel_1 + RAY_ANGLE_OFFSET * i)
            x1, y1 = player_1.center
            ray_distances[i] = RAY_LENGTH

            for j in range(RAY_LENGTH):
                end_x = int(x1 + j * math.cos(angle))
                end_y = int(y1 + j * math.sin(angle))

                if end_x < 0 or end_x >= ww or end_y < 0 or end_y >= wh:
                    break

                if fenster.get_at((end_x, end_y)) == c_fence:
                    ray_distances[i] = j
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
                zaehler += 1
                player_1.left = 100
                player_1.top = 165
                winkel_1 = 0

                if zaehler >= len(strecken):
                    zaehler = 0

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

    ray_info = ", ".join([f"Ray {i + 1}: {round(dist, 2)}" for i, dist in enumerate(ray_distances)])
    data_to_send = f"Speed: {round(bew_zaehler_1, 2)}, Acceleration: {acceleration}, Brake: {brake}, " \
                   f"Angle: {round(winkel_1, 2)}, Lose: {lose}, Win: {win}, {ray_info}"
    client_socket.sendall(data_to_send.encode())

    # Check if there's data available to be received
    readable, _, _ = select.select([client_socket], [], [], 0)
    if readable:
        data = client_socket.recv(1024)
        if data:
            action = int(data.decode())
            print("Received action from server:", action)

            # Based on the received action, perform the corresponding action in the game
            if action == 0:  # Forward
                pressed_1 = True
                pressed_1_b = False
                pressed_1_l = False
                pressed_1_r = False
            elif action == 1 :  # Backward
                pressed_1 = False
                pressed_1_b = True
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

    fenster.fill((0, 0, 0))
    fenster.blit(strecken[zaehler], (0, 0))

    if count_destr_1 == 0:
        try:
            if not fenster.get_at((player_1.left + 10, player_1.top + 10)) == c_straße:
                if bew_zaehler_1 > 3:
                    bew_zaehler_1 = 2
                if bew_zaehler_1 < -3:
                    bew_zaehler_1 = -2

            if fenster.get_at((player_1.left + 10, player_1.top + 10)) == c_fence:
                destroy_1 = 1
                lose = 1

            if fenster.get_at((player_1.left + 10, player_1.top + 10)) == c_finish:
                destroy_1 = 1
                win = 1

        except:
            destroy_1 = 1

        if destroy_1 == 0:
            fenster.blit(image_1_neu, player_1)

    else:
        fenster.blit(explosion, player_1)

    # Display current speed, acceleration, brake, angle
    font = pygame.font.Font(None, 36)
    text_speed = font.render(f"Speed: {round(bew_zaehler_1, 2)}", True, blue)
    text_acceleration = font.render(f"Acceleration: {acceleration}", True, blue)
    text_brake = font.render(f"Brake: {brake}", True, blue)
    text_angle = font.render(f"Angle: {round(winkel_1, 2)}", True, blue)
    text_lose = font.render(f"Lose: {round(lose, 2)}", True, blue)
    text_finish = font.render(f"Finish distance: {round(finish_distance, 2)}", True, blue)

    fenster.blit(text_speed, (10, 10))
    fenster.blit(text_acceleration, (10, 50))
    fenster.blit(text_brake, (10, 90))
    fenster.blit(text_angle, (10, 130))
    fenster.blit(text_lose, (10, 170))
    fenster.blit(text_lose, (10, 210))


    # Display ray distances and angles
    for i in range(RAY_COUNT):
        angle = math.radians(winkel_1 + RAY_ANGLE_OFFSET * i)
        end_x = player_1.centerx + ray_distances[i] * math.cos(angle)
        end_y = player_1.centery + ray_distances[i] * math.sin(angle)
        pygame.draw.line(fenster, blue, player_1.center, (end_x, end_y), 2)

        # Display ray info
        ray_info = font.render(f"Ray {i + 1}: Angle = {round(winkel_1 + RAY_ANGLE_OFFSET * i, 2)}, "
                               f"Distance = {ray_distances[i]}", True, blue)
        fenster.blit(ray_info, (10, 250 + 30 * i))

    if destroy_1 == 1:
        fenster.blit(explosion, player_1)
        pygame.display.update()
        destroy_1 = 0
        winkel_1 = 0
        lose = 0
        win = 0
        count_destr_1 = 25

    pygame.display.update()
    clock.tick(fps)

pygame.quit()
