def config_params():
    params = {}
    # GPU index to use
    params["gpu_num"] = 0
    # Tensorflow seed
    params["seed"] = 1
    # Sionna scene to use for ray tracing
    params["scene_path"] = "scenes/office_v1/office_v1.xml"
    # Tx positions
    params["tx_pos"] = [13.0, 0.0, 2.0]
    # Path to the rx position file
    params["rx_pos_path"] = params["scene_path"][:-4] + "_rx.csv"
    # Number of samples used for tracing
    params["num_samples"] = int(1e5)
    # Maximum depth used for tracing
    params["max_depth"] = 3
    # Enables LoS when tracing
    params["los"] = True
    # Enables reflection when tracing
    params["reflection"] = True
    # Enables diffraction when tracing
    params["diffraction"] = True
    # Enables edge diffraction when tracing
    params["edge_diffraction"] = False
    # Enables scattering when tracing
    params["scattering"] = True
    params["scattering_coefficient"] = 0.3 * params["scattering"]
    params["xpd_coefficient"] = 0.2 * params["scattering"]
    params["scattering_pattern"] = {"alpha_r": 5, "alpha_i": 8, "lambda_": 0.8}

    # Probability to keep a scattered paths when tracing
    params["scat_keep_prob"] = 0.001
    # Filename of the dataset of traced paths to create
    params["traced_paths_dataset"] = "office_v1"
    # Size of the dataset of traced paths
    # Set to -1 to match the datset of measurements
    params["traced_paths_dataset_size"] = 30000
    # Delete the raw dataset once post-processed?
    params["delete_raw_dataset"] = True
    # Folder where to save the dataset
    params["traced_paths_dataset_folder"] = "data/traced_paths"
    return params
