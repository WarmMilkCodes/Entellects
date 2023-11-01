import pygame, pygame.font, random, math, torch
from .entities import entellect, food, learning
from .environment import timesystem
from .utils import interpolate_color, get_background_color
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Initialize the environment
pygame.init()
font = pygame.font.SysFont(None, 36)
previous_ticks = pygame.time.get_ticks()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Colors
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DAY_COLOR = (100, 200, 100) # Green
NIGHT_COLOR = (20, 40, 20) # Darker green
        
# Generate initial food sources
foods = [Food(random.randint(0, screen_width), random.randint(0, screen_height - 100)) for _ in range(10)]

# Create an Entellect
entellects = [Entellect(screen_width//2, screen_height//2),
              Entellect(screen_width//3, screen_height//3)]

running = True
clock = pygame.time.Clock()

# Initialize time system
time_system = TimeSystem()

while running:
    x_pos = int(ent.x.item())
    y_pos = int(ent.y.item())
    current_ticks = pygame.time.get_ticks()
    delta_time = (current_ticks - previous_ticks) / 1000
    previous_ticks = current_ticks

    time_system.update(delta_time)

    # Retrieve time of day and current hour
    current_hour = time_system.current_time % 24
    time_of_day = time_system.get_time_of_day()

    # Get appropriate background color based on time of day
    bg_color = get_background_color(time_of_day, current_hour)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update Entellects
    for ent in entellects:
        ent.update(delta_time)

        # Check if Entellect is close to any food source
        for food in foods:
            if food.is_eaten_by(ent):
                ent.eat(food)
                foods.remove(food)
                break

    # Draw background (grass and water)
    screen.fill(bg_color)
    pygame.draw.rect(screen, BLUE, (0, screen_height - 100, screen_width, 100))

    # Draw food
    for food in foods:
        food.draw()

    # Draw Entellects
    for ent in entellects:
        ent.draw()

        # Check for hover
        if ent.is_hovered(pygame.mouse.get_pos()):
            # Render the vitals
            energy_text = font.render(f"Energy: {ent.energy:.2f}", True, (WHITE))
            hydration_text = font.render(f"Hydration: {ent.hydration:.2f}", True, (WHITE))
            screen.blit(energy_text, (x_pos, y_pos - 10))
            screen.blit(hydration_text, (x_pos, y_pos - 35))

    # Render in-game date and time
    in_game_days = int(time_system.current_time // 24)
    in_game_hours = int(time_system.current_time % 24)
    time_text = f"Day {in_game_days}, {in_game_hours:02d}:00 {time_system.get_time_of_day().capitalize()}, {time_system.get_season().capitalize()}"
    time_surface = font.render(time_text, True, (0, 0, 0))
    screen.blit(time_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)  # Cap the frame rate at 60 FPS


pygame.quit()
