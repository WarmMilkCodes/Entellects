import pygame
import pygame.font
import random
import math
from .entities import entellect, food
import torch
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


def interpolate_color(color1, color2, factor):
    """
    Interpolate between two colors.
    Factor is between 0 and 1, where 0 gives color1, 1 gives color2 and 0.5 gives an evenly mixed color.
    """
    r = color1[0] + factor * (color2[0] - color1[0])
    g = color1[1] + factor * (color2[1] - color1[1])
    b = color1[2] + factor * (color2[2] - color1[2])
    return int(r), int(g), int(b)

def get_background_color(time_of_day, current_hour):
    if time_of_day == "morning" or time_of_day == "afternoon":
        return DAY_COLOR
    elif time_of_day == "evening":
        # Transition from day to night during evening
        factor = (current_hour - 18) / 6  # Assuming evening is from 18 to 24
        return interpolate_color(DAY_COLOR, NIGHT_COLOR, factor)
    else:  # night
        return NIGHT_COLOR

# Reinforcement Learning
class QLearning:
    def __init__(self, brain, learning_rate=0.01, discount_factor=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory = []
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def get_target_q_value(self, reward, next_state):
        # Bellman Equation
        target_q_value = reward + self.discount_factor * torch.max(self.brain(next_state))
        return target_q_value

    def replay(self):
        for state, action, reward, next_state in self.memory:
            predicted_q_value = self.brain(inputs)
            target_q_value = target_q_value.unsqueeze(0).expand_as(predicted_q_value)
            loss = F.mse_loss(predicted_q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory.clear()

# Neural Networking
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Input: relative x and y to the closest food
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)  # Output: movement direction in x and y

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   


# Time class
class TimeSystem:
    def __init__(self, initial_time=0, time_scale=24):
        self.current_time = initial_time
        self.time_scale = time_scale

    def update(self, delta_time):
        # Update the in-game time based on the elapsed real-world time.
        self.current_time += delta_time / 60 * self.time_scale

    def get_time_of_day(self):
        # Return the current time of day: morning, afternoon, evening, night
        hour = self.current_time % 24
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"
        
    def get_season(self):
        # Return the current season: spring, summer, fall, winter
        day_of_year = (self.current_time // 24) % 365
        if 0 <= day_of_year < 91:
            return "spring"
        elif 91 <= day_of_year < 182:
            return "summer"
        elif 182 <= day_of_year < 273:
            return "fall"
        else:
            return "winter"
        
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
