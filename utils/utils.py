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
