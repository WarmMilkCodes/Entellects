class Resource:
  def __init__(self, resource_type, quantity, x, y):
    self.resource_type = resource_type
    self.quantity = quantity
    self.x = x
    self.y = y

  def is_collected_by(self, entellect):
    # Logic to determine if Entellect is close enough to collect
    pass

  def draw(self, screen):
    # Render the resource on screen based on its type
    pass

  def collected_with(self):
    return "Hands" # Default 

class Water(Resource):
  def __init__(self, quantity):
    super().__init__(resource_type="Survival", quantity=quantity, x=x, y=y)
    # other unique properties of water

  def collected_with(self):
    return "Bucket"

class Wood(Resource):
  def __init__(self, quantity):
    super().__init__(resource_type="Building", quantity=quantity, x=x, y=y)
    # other unique properties

  def collected_with(self):
    return "Axe"

class Stone(Resource):
  def __init__(self, quantity):
    super().__init__(resource_type="Building", quantity=quantity, x=x, y=y)
    # other unique properties

  def collected_with(self):
    return "Pickaxe" 

    
