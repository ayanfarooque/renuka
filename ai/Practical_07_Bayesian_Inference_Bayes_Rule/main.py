rain = 0.2
cloudy_rain = 0.8
cloudy_no_rain = 0.1

cloudy = rain * cloudy_rain + (1 - rain) * cloudy_no_rain
result = rain * cloudy_rain / cloudy

print("P(Rain | Cloudy):", round(result, 4))
