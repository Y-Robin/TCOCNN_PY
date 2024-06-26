import numpy as np

def data_ds_fixed(data_train, target_train, data_test, target_test):
    """
    DataDS performs direct standardization between master and slave sensors.
    This is a common task for calibration transfer of MOSGasSensors.

    data_train: Training data of the master sensor.
    target_train: Target values for the training data.
    data_test: Test data of the slave sensor.
    target_test: Target values for the test data.

    Returns:
    data_trans: Dictionary containing standardized data of the slave sensor.
    """
    data_trans = {
        'train': [],
        'val': [],
        'test': []
    }

    # TransformationVar
    F = [None] * data_train.shape[1]


    # If multiple parent sensors are available
    multiplier_UGM = target_train.shape[0]/target_test['train'].shape[0]
    if int(multiplier_UGM) != multiplier_UGM:
        print("Warning, multiple UGMs have same target might cause problem")

    slave_sensor_array = data_test['train']

    for i in range(int(multiplier_UGM)-1):
        slave_sensor_array = np.concatenate((slave_sensor_array,data_test['train']),axis=0)

    # Calculate for every subsensor the C parameter
    for i in range(data_test['train'].shape[1]):
        F[i] = np.dot(data_train[:,i,:,0].T, np.linalg.pinv(slave_sensor_array[:,i,:,0].T))

    data_trans_train = np.zeros((data_test['train'].shape))
    data_trans_val = np.zeros((data_test['val'].shape))
    data_trans_test = np.zeros((data_test['test'].shape))

    # Transform Data
    for i in range(data_test['train'].shape[1]):
        for k in range(data_test['train'].shape[0]):
            data_trans_train[k,i,...] = np.dot(F[i],data_test['train'][k,i,...])
        for k in range(data_test['val'].shape[0]):  
            data_trans_val[k,i,...]  = np.dot(F[i],data_test['val'][k,i,...])
        for k in range(data_test['test'].shape[0]):  
            data_trans_test[k,i,...]  = np.dot(F[i],data_test['test'][k,i,...])

    data_trans['train'] = data_trans_train
    data_trans['val'] = data_trans_val
    data_trans['test'] = data_trans_test

    return data_trans
