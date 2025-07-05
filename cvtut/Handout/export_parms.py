import yaml

parameters = {
    'brightness_threshold': 60,
    'light_min_area': 10,
    'light_max_angle': 45.0,
    'light_min_size': 3.0,
    'light_contour_min_solidity': 0.3,
    'light_max_ratio': 0.6,
    'light_color_detect_extend_ratio': 1.1,
    'light_max_angle_diff': 7.0,
    'light_max_length_diff_ratio': 0.3,
    'light_max_y_diff_ratio': 2.0,
    'light_min_x_diff_ratio': 0.5,
    'armor_min_aspect_ratio': 1.0,
    'armor_max_aspect_ratio': 5.0
}

with open('parameters.yaml', 'w') as file:
    yaml.dump(parameters, file)