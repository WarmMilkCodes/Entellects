import pygame
import pygame.font
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

# Food class
class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 5
        self.energy_value = 10

    def draw(self):
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.size)

    def is_eaten_by(self, entellect):
        distance = math.sqrt((self.x - entellect.x) ** 2 + (self.y - entellect.y) ** 2)
        return distance <= (self.size + entellect.size)

# Entellect class
class Entellect:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 10
        self.energy = 100
        self.hydration = 100
        self.brain = Brain()

    def choose_action(self):
        inputs = self.get_state()
        outputs = self.brain(inputs)
        return outputs

    def draw(self):
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.size)
        
    def eat(self, food):
        self.energy += food.energy_value
        self.energy = min(self.energy, 100)
    
    def get_state(self):
       pass

    def get_reward(self):
        reward = 0
        if self.is_near_food():
            reward += 10  # Example reward value
        if self.is_near_water():
            reward += 5
        if self.is_off_screen():
            reward -= 20

        return reward

    def update(self, delta_time):
        # Decrease energy over time
        energy_depletion_rate = 0.02315
        self.energy -= energy_depletion_rate * delta_time
        self.energy = max(self.energy, 0)

        # Decrease hydration over time
        hydration_depletion_rate = 0.05
        self.hydration -= hydration_depletion_rate * delta_time
        self.hydration = max(self.hydration, 0)

        seek_water = self.hydration < 30 and self.hydration < self.energy

        if seek_water:
            relative_x = 0
            relative_y = (screen_height - 100) - self.y
        else:
            closest_food = min(foods, key=lambda food: (food.x - self.x)**2 + (food.y - self.y)**2)
            relative_x = closest_food.x - self.x
            relative_y = closest_food.y - self.y 

        # Feed the relative position to the neural network
        inputs = torch.tensor([self.x / screen_width, self.y / screen_height, self.energy / 100.0]).float().unsqueeze(0)
        outputs = self.brain(inputs)
        
        # Use the outputs as velocity
        self.vx = outputs[0, 0].item() * 15
        self.vy = outputs[0, 1].item() * 15

        # Update position
        self.x += self.vx * delta_time
        self.y += self.vy * delta_time

        # Constrain to screen boundaries
        self.x = max(min(self.x, screen_width - self.size), self.size)
        self.y = max(min(self.y, screen_height - self.size - 100), self.size)  # -100 to account for the water's edge

        if self.energy == 0:
            # Entellect dies
            pass

        # Check if near water
        if screen_height - 100 <= self.y <= screen_height - 85:
            self.hydration += 15

        # RL Logic
        state = self.get_state()
        action = self.choose_action()
        # Apply the action
        self.apply_action(action)
        reward = self.get_reward()
        next_state = self.get_state()

        self.train_nn(state, action, reward, next_state)

    def train_nn(self, state, action, reward, next_state):
        pass

    def is_hovered(self, mouse_pos):
        distance = math.sqrt((self.x - mouse_pos[0]) ** 2 + (self.y - mouse_pos[1]) ** 2)
        return distance <= self.size

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
            screen.blit(energy_text, (ent.x + 15, ent.y - 10))
            screen.blit(hydration_text, (ent.x + 15, ent.y - 35))

    # Render in-game date and time
    in_game_days = int(time_system.current_time // 24)
    in_game_hours = int(time_system.current_time % 24)
    time_text = f"Day {in_game_days}, {in_game_hours:02d}:00 {time_system.get_time_of_day().capitalize()}, {time_system.get_season().capitalize()}"
    time_surface = font.render(time_text, True, (0, 0, 0))
    screen.blit(time_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)  # Cap the frame rate at 60 FPS


pygame.quit()
