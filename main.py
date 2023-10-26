import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random
import math

class EntityNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EntityNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=0)

class TimeSystem:
    def __init__(self, initial_time=0, time_scale=24):
        self.current_time = initial_time # time in hours
        self.time_scale = time_scale # how many simulation hours pass per real-world hour

    def update(self, delta_time):
        self.current_time += delta_time / 3600 * self.time_scale

    def get_time_of_day(self):
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
        day_of_year = (self.current_time // 24) % 365
        if 0 <= day_of_year < 91:
            return "spring"
        elif 91 <= day_of_year < 182:
            return "summer"
        elif 182 <= day_of_year < 273:
            return "fall"
        else:
            return "winter"
        
class Entity:
    def __init__(self, x, y, gender=None, name=None):
        self.x = x
        self.y = y
        self.energy = 100
        self.age = 0
        self.gender = gender or random.choice(['male', 'female'])
        self.neural_network = EntityNN(input_size=4, output_size=5)
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)
        self.memory = []
        self.name = name or self.generate_name()
        self.reproduction_cooldown = 0 # attribute to manage reproduction cooldown
        self.resources = {'wood': 0} # Add other resource types
        self.shelter = None
        self.vx = 0 # velocity in x-direction
        self.vy = 0 # velocity in y-direction
        self.speed = 2 # speed at which Entellect moves
        self.velocity_decay = 0.95 # factor by which velocity decays each update
        self.relationships = {}

    def adjust_relationship(self, other, value):
        """Adjust relationship value with another entity."""
        if other not in self.relationships:
            self.relationships[other] = 0
        self.relationships[other] += value
        # Keeping relationship values between -100 and 100
        self.relationships[other] = max(-100, min(100, self.relationships[other]))
    
    def check_mortality(self, environmental_factor=1.0):
        # Energy-based mortality
        if self.energy <= 0:
            return True

        # Environmental factor mortality (eg, harsh winter)
        if self.energy < 10 and random.random() < 0.01 * environmental_factor:
            return True

        # Random chance mortality
        if random.random() < 0.001: # 0.01% chance every tick
            return True
        
        return False

    def gather_resources(self, resource):
        if self.nearby(resource):
            self.resource[resource.type] += resource.gather()

    def construct_shelter(self):
        if self.resources['wood'] >= 10: # example condition
            self.shelter = Shelter(quality=self.resources['wood'])
            self.resources['wood'] -= 10

    def seek_shelter(self):
        if self.shelter and self.shelter.quality > 0:
            self.in_shelter = True
        else:
            # Find nearest shelter or community
            pass
        
    def generate_name(self):
        syllables = ['ka', 'ri', 'to', 'na', 'lu', 'mi']
        name_length = random.randint(2, 3)
        return ''.join(random.choice(syllables) for _ in range(name_length))

    def can_reproduce_with(self, other_entity):
        # Basic criteria for reproduction
        if self.gender != other_entity.gender and self.age > 10 and other_entity.age > 10:
            return True
        return False

    def reproduce(self, other_entity):
        if self.can_reproduce_with(other_entity) and self.reproduction_cooldown <= 0 and other_entity.reproduction_cooldown <= 0:
            # Determine offspring properties based on parents
            child_x = (self.x + other_entity.x) // 2
            child_y = (self.y + other_entity.y) // 2
            child = Entity(child_x, child_y)
            child_name = self.name[:2] + other_entity.name[-2:]
            child = Entity(child_x, child_y, name=child_name)
            self.reproduction_cooldown = 24 # 24 hour cooldown after reproducing
            other_entity.reproduction_cooldown = 24
            return child
        
    def get_state(self, food_sources, entities):
        nearest_food_dist = min([((self.x - fx)**2 + (self.y -fy)**2)**0.5 for fx, fy in food_sources])

        # Find the nearest entity and its details
        nearest_entity = min(entities, key=lambda e: ((self.x - e.x)**2 + (self.y - e.y)**2)**0.5 if e != self else float('inf'))
        nearest_entity_dist = ((self.x - nearest_entity.x)**2 + (self.y - nearest_entity.y)**2)**0.5
        relationship_value = self.relationships.get(nearest_entity, 0)
        return torch.tensor([self.x, self.y, self.energy, nearest_food_dist, nearest_entity_dist, nearest_entity.age, relationship_value], dtype=torch.float32)
        
    def decide_action(self, food_sources, entities, epsilon=0.1):
        state = self.get_state(food_sources, entities)
        return self.epsilon_greedy_action(state, epsilon)
    
    def perform_action(self, action, angle, food_sources, shelters, entities):
        if action == 0: # move in a direction based on angle
            self.vx = self.speed * math.cos(angle)
            self.vy = self.speed * math.sin(angle)
            self.energy -= 0.2
            
        elif action == 4: # interact
            self.energy -= 0.5
            
            # Interact with food sources
            for food in food_sources:
                if abs(self.x - food[0]) < 5 and abs(self.y - food[1]) < 5: # 5 is interaction distance
                    self.energy += 10 # Increase energy for consuming food source
                    food_sources.remove(food)
                    break
            
            # Interact with shelters
            if not self.shelter:
                for shelter in shelters:
                    if abs(self.x - shelter.x) < 10 and abs(self.y - shelter.y) < 10: # 10 is interaction distance
                        if not shelter.occupied:
                            self.shelter = shelter
                            self.shelter.occupied = True
                            break

            # If no shelter and enough resources, construct one
            if not self.shelter and self.resources['wood'] >= 10:
                self.resources['wood'] -= 10
                self.shelter = Shelter(self.x, self.y)
                shelters.append(self.shelter)

            # Interact with other entities
            for other in entities:
                if other != self and abs(self.x - other.x) < 5 and abs(self.y - other.y) < 5: # 5 is interaction distance
                    relationship_value = self.relationships.get(other, 0)
                    
                    # If they have a good relationship
                    if relationship_value > 5:
                        interaction_type = 'communicate'
                    # If they have a neutral relationship
                    elif -5 <= relationship_value <= 5:
                        interaction_type = 'exchange'
                    # If they have a bad relationship
                    else:
                        interaction_type = 'conflict'

                    if interaction_type == 'communicate':
                        # Strengthen relationship slightly
                        self.adjust_relationship(other, 2)
                        # Learn about a distant food source or something similar
                        # (e.g., update an internal map or knowledge base)

                    elif interaction_type == 'exchange':
                        # Entities exchange resources based on their needs and availability
                        if self.resources['wood'] > other.resources['wood'] + 1:
                            self.resources['wood'] -= 1
                            other.resources['wood'] += 1
                        # Improve relationship status
                        self.adjust_relationship(other, 3)

                    elif interaction_type == 'conflict':
                        # Decrease in energy due to conflict
                        self.energy -= 5
                        other.energy -= 5
                        # Worsen relationship status drastically
                        self.adjust_relationship(other, -10)

        # Apply the velocity to the position and decay the velocity
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.velocity_decay
        self.vy *= self.velocity_decay
        
    def receive_reward(self, action, food_sources):
        # Default negative reward for energy expenditure
        reward = -1

        # If the entity interacts
        if action == 4:
            interaction_reward = 0

            # Check for food within interaction range
            for food in food_sources:
                if abs(self.x - food[0]) < 5 and abs(self.y - food[1]) < 5:
                    interaction_reward += 10 # Positive reward for finding food
                    self.energy += 10 # Increase energy for consuming food
                    food_sources.remove(food)
                    break
            
            # If no food was found within interaction range, slightly penalize
            if interaction_reward == 0:
                reward -= 2

        # If the entity moves, check if it's moving towards food when energy is low
        elif self.energy < 50:
            nearest_food_dist = min([((self.x - fx)**2 + (self.y - fy)**2)**0.5 for fx, fy in food_sources])
            new_x, new_y = self.x, self.y
        if action == 0:  # Move Up
            new_y -= 1
        elif action == 1:  # Move Down
            new_y += 1
        elif action == 2:  # Move Left
            new_x -= 1
        elif action == 3:  # Move Right
            new_x += 1
        new_nearest_food_dist = min([((new_x - fx)**2 + (new_y - fy)**2)**0.5 for fx, fy in food_sources])
        
        # If moving closer to food, slightly reward the entity
        if new_nearest_food_dist < nearest_food_dist:
            reward += 2
        else:
            reward -= 2  # Penalize if moving away from food

        return reward

    def interact_with_entity(self, other):
        # Check the relationship value
        relationship_value = self.relationships.get(other, 0)
        
        # If they have a good relationship
        if relationship_value > 5:
            interaction_type = 'communicate'
        # If they have a neutral relationship
        elif -5 <= relationship_value <= 5:
            interaction_type = 'exchange'
        # If they have a bad relationship
        else:
            interaction_type = 'conflict'
            
        if interaction_type == 'communicate':
            # Improve relationship status slightly
            self.relationships[other] = relationship_value + 1
            other.relationships[self] = other.relationships.get(self, 0) + 1

        elif interaction_type == 'exchange':
            # Exchange resources or knowledge
            # Example: if one has more wood, it gives some to the other
            if self.resources['wood'] > other.resources['wood'] + 1:
                self.resources['wood'] -= 1
                other.resources['wood'] += 1
            else:
                # They exchange knowledge or other resources
                pass
            # Improve relationship status
            self.relationships[other] = relationship_value + 2
            other.relationships[self] = other.relationships.get(self, 0) + 2

        elif interaction_type == 'conflict':
            # Worsen relationship status drastically
            self.relationships[other] = relationship_value - 5
            other.relationships[self] = other.relationships.get(self, 0) - 5
    
    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 4), random.uniform(0, 2*math.pi) # return both action and random angle
        else:
            with torch.no_grad():
                q_values = self.neural_network(state)
            return q_values.argmax().item(), random.uniform(0, 2*math.pi)
        
    def store_experiences(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def train(self, batch_size=32, gamma=0.99):
        # Check if enough experiences are available
        if len(self.memory) < batch_size:
            return

        # Sample experiences from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)

        # Compute predicted Q-values
        predicted_q_values = self.neural_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.neural_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(predicted_q_values, target_q_values)

        # Optimize neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def store_experience(self, state, action, reward):
        self.memory.append((state, action, reward))

class Shelter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.quality = 10 # Quality of the shelter which can degrade over time
        self.occupied = False

# Create list for shelters
shelters = []
    
# Initialize the time system
time_system = TimeSystem()

# Initialize the environment
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Load font for text display
font = pygame.font.SysFont(None, 36) # default font, size 36

# Create entities and food sources
entities = [Entity(random.randint(0, 800), random.randint(0, 600)) for _ in range(2)]
food_sources = [(random.randint(0, 800), random.randint(0, 600)) for _ in range(10)]

# Initialize previous_ticks
previous_ticks = pygame.time.get_ticks()

# Simulation loop
running = True
delta_time = 1.0
epsilon = 1.0 # For epsilon-greedy exploration
epsilon_decay = 0.995 # Decay rate for epsilon

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    

    # Calculate delta_time (time passed since last frame)
    current_ticks = pygame.time.get_ticks()
    delta_time = (current_ticks - previous_ticks) / 1000.0 # seconds
    previous_ticks = current_ticks                  

    # Update time system
    time_system.update(delta_time * 60) # 1 minute real time = 1 hour in sim
    current_season = time_system.get_season()
    current_time_of_day = time_system.get_time_of_day()

    # Print current time and season (for testing)
    print(f"Current time: {time_system.current_time:.2f} hours, Season: {time_system.get_season()}")

    # Adjust environmental factor based on season and time of day
    environmental_factor = 1.0
    if current_season == "winter" and current_time_of_day == "night":
        environmental_factor = 1.5
    
    # Update entities
    for entity in entities:
        # Entity ages with the passage of time
        entity.age += delta_time / 3600.0 # Conver seconds to hours
        action, angle = entity.decide_action(food_sources, epsilon)
        entity.perform_action(action, angle, food_sources, shelters)
        reward = entity.receive_reward(action, food_sources)
        next_state = entity.get_state(food_sources)
        entity.store_experience(entity.get_state(food_sources), action, reward, next_state)
        entity.train()
        if entity.check_mortality(environmental_factor):
            entities.remove(entity)
        # Decrement reproduction cooldown for each entity
        if entity.reproduction_cooldown > 0:
            entity.reproduction_cooldown -= delta_time / 3600.0

    # Seasonal dynamics (ex. adjusting food regen rate)
    if current_season in ['spring', 'summer']:
        if random.random() < 0.05: # 5% chance for a new food source to appear
            food_sources.append((random.randint(0, 800), random.randint(0, 600)))
    elif current_season in ['fall', 'winter']:
        if random.random() < 0.02: # 2% chance for a new food source to appear
            food_sources.append((random.randint(0, 800), random.randint(0, 600)))
        
    # Check for entities in proximity and allow them to reproduce
    new_offsprings = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j and abs(entity1.x - entity2.x) < 5 and abs(entity1.y - entity2.y) < 5:
                child = entity1.reproduce(entity2)
                if child:
                    new_offsprings.append(child)
    entities.extend(new_offsprings)

    # Decay epsilon
    epsilon *= epsilon_decay

    # Render entities and food sources on the screen
    screen.fill((200, 200, 200))
    for entity in entities:
        pygame.draw.circle(screen, (0, 0, 0), (entity.x, entity.y), 5)
        # Render age and energy for each entity
        age_text = font.render(f"Age: {entity.age:.2f}", True, (0, 0, 0))
        energy_text = font.render(f"Energy: {entity.energy}", True, (0, 0, 0))
        screen.blit(age_text, (entity.x - 10, entity.y - 10))
        screen.blit(energy_text, (entity.x - 10, entity.y))
    for food in food_sources:
        pygame.draw.circle(screen, (0, 255, 0), food, 3)
    pygame.display.flip()

    # Render in-game date and time
    in_game_days = int(time_system.current_time // 24)
    in_game_hours = int(time_system.current_time % 24)
    time_text = f"Day {in_game_days}, {in_game_hours:02d}:00 {time_system.get_time_of_day().capitalize()}, {time_system.get_season().capitalize()}"
    time_surface = font.render(time_text, True, (0, 0, 0))
    screen.blit(time_surface, (10, 10))

pygame.quit()

