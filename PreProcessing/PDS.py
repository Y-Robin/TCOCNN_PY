import numpy as np

def data_pds_fixed(data_train, target_train, data_test, target_test, piece_size):
    """
    DataDS performs direct standardization between master and slave sensors.
    This is a common task for calibration transfer of MOSGasSensors.

    data_train: Training data of the master sensor.
    target_train: Target values for the training data.
    data_test: Test data of the slave sensor.
    target_test: Target values for the test data.
    piece_size: Number of samples to use within the third dimension to calculate the F values.

    Returns:
    data_trans: Dictionary containing standardized data of the slave sensor.
    """
    data_trans = {
        'train': np.zeros_like(data_test['train']),
        'val': np.zeros_like(data_test['val']),
        'test': np.zeros_like(data_test['test'])
    }

    # If multiple parent sensors are available
    multiplier_UGM = target_train.shape[0] / target_test['train'].shape[0]
    if int(multiplier_UGM) != multiplier_UGM:
        print("Warning, multiple UGMs have same target might cause problem")

    # Expand the slave sensor array
    slave_sensor_array = data_test['train']
    for i in range(int(multiplier_UGM) - 1):
        slave_sensor_array = np.concatenate((slave_sensor_array, data_test['train']), axis=0)

    # Calculate F values piecewise
    num_pieces = data_train.shape[2] // piece_size
    F_pieces = np.zeros((data_train.shape[1], num_pieces, piece_size, piece_size))

    for i in range(data_train.shape[1]):  # Iterate over subsensors
        for p in range(num_pieces):
            start = p * piece_size
            end = (p + 1) * piece_size
            F_pieces[i, p] = np.dot(
                data_train[:, i, start:end, 0].T,
                np.linalg.pinv(slave_sensor_array[:, i, start:end, 0].T)
            )

    # Transform data piecewise
    for i in range(data_train.shape[1]):  # Iterate over subsensors
        for p in range(num_pieces):
            start = p * piece_size
            end = (p + 1) * piece_size
            F = F_pieces[i, p]

            for k in range(data_test['train'].shape[0]):
                data_trans['train'][k, i, start:end, 0] = np.dot(F, data_test['train'][k, i, start:end, 0])

            for k in range(data_test['val'].shape[0]):
                data_trans['val'][k, i, start:end, 0] = np.dot(F, data_test['val'][k, i, start:end, 0])

            for k in range(data_test['test'].shape[0]):
                data_trans['test'][k, i, start:end, 0] = np.dot(F, data_test['test'][k, i, start:end, 0])

    return data_trans