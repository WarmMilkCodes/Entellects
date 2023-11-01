
class Entellect:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 10
        self.energy = 100
        self.hydration = 100
        self.brain = Brain()
        self.brain_optimizer = torch.optim.Adam(self.brain.paramaters(), lr=0.001)
        self.q_learning = QLearning(self.brain)

    def choose_action(self):
        inputs = self.get_state()
        outputs = self.brain(inputs)
        return outputs

    def draw(self):
        x_pos = int(self.x.item())
        y_post = int(self.y.item())
        pygame.draw.circle(screen, WHITE, (x_pos, y_pos), self.size)

    def eat(self, food):
        self.energy += food.energy_value
        self.energy = min(self.energy, 100)

    def get_state(self):
       return torch.tensor([self.x / screen_width, self.y / screen_height, self.energy / 100.0]).float().unsqueeze(0)

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

        # Constrain to screen boundaries with a margin
        MARGIN = 5
        self.x = max(min(self.x, screen_width + MARGIN), -MARGIN)
        self.y = max(min(self.y, screen_height + MARGIN), -MARGIN)  # -100 to account for the water's edge


        if self.energy == 0:
            # Entellect dies
            pass

        # Check if near water
        if screen_height - 100 - MARGIN <= self.y <= screen_height - 85 + MARGIN:
            self.hydration += 15

        # RL Logic
        state = self.get_state()
        action = self.choose_action()
        self.apply_action(action)
        reward = self.get_reward()
        next_state = self.get_state()
        self.q_learning.remember(state, action, reward, next_state)
        self.q_learning.replay()

        self.train_nn(state, action, reward, next_state)

    def train_nn(self, state, action, reward, next_state):
        pass

    def is_hovered(self, mouse_pos):
        distance = math.sqrt((self.x - mouse_pos[0]) ** 2 + (self.y - mouse_pos[1]) ** 2)
        return distance <= self.size

    def apply_action(self, action):
        # Assuming the action contains velocities or changes in x and y positions
        dx, dy = action[0]
        self.x += dx
        self.y += dy
        # Ensure the entellect remains within the screen boundaries
        self.x = max(min(self.x, screen_width - self.size), self.size)
        self.y = max(min(self.y, screen_height - self.size - 100), self.size)

    def is_near_food(self):
        for food in foods:
            distance = math.sqrt((self.x - food.x) ** 2 + (self.y - food.y) ** 2)
            if distance <= 30:
                return True
        return False

    def is_near_water(self):
        return screen_height - 100 <= self.y <= screen_height - 90

    def is_off_screen(self):
        return self.x < 0 or self.x > screen_width or self.y < 0 or self.y > screen_height
