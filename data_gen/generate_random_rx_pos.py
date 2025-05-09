import pandas as pd
import numpy as np


def generate_positions(num_data, x_range, y_range, z_range, pos_avoid, min_distance):
    pos_avoid = np.asarray(pos_avoid, dtype=float)
    min_distance = np.asarray(min_distance, dtype=float)
    positions = []

    while len(positions) < num_data:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])

        # Check if the generated position is at least `min_distance` away from all positions in pos_avoid
        if np.all(
            np.sqrt(
                (x - pos_avoid[:, 0]) ** 2
                + (y - pos_avoid[:, 1]) ** 2
                + (z - pos_avoid[:, 2]) ** 2
            )
            >= min_distance
        ):
            positions.append([x, y, z])

    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    return df


# Example usage with pandas and fixed seed
scene_name = "office_v1"
save_path = f"scenes/{scene_name}/{scene_name}_rx.csv"
num_data = 100000
x_range = [-15, 15]
y_range = [-15, 15]
z_range = [0.5, 2.5]
pos_avoid = [[0.0, 0.0, 0.0]]
min_distance = [0.0]

df = generate_positions(num_data, x_range, y_range, z_range, pos_avoid, min_distance)
df.to_csv(save_path, index=False)
