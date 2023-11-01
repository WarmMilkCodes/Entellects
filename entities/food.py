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
