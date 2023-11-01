
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
