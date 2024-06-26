import numpy as np
import h5py
import scipy.io as sio
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_Data_Full(load_struct):
    method_names = ["","Split", "Transfer", "Drift", "Measurement", "Fieldtests", "SpecificUGMs"]
    
    # Extract Struct
    file_name_data_all = load_struct['fileNameDataAll']
    file_name_target_all = load_struct['fileNameTargetAll']
    target_gas = load_struct['targetGas']
    # Used Method
    load_method = load_struct['loadMethod']
    
    # Expected Input Dimensions
    data_size = load_struct['dataSize']
    num_sub_sensors = data_size[0]
    num_samples = data_size[1]

    # Random Flag
    random_flag = load_struct['randomFlag']

    # Normalize Input 
    norm_flag = load_struct['normFlag']
    
    occlusion_flag = load_struct['OcclusionFlag']
    
    # Empty Storage for Output
    target = {'train': np.empty((0,)), 'val': np.empty((0,)), 'test': np.empty((0,)), 'remain': np.empty((0,))}
    data = {'train': np.empty((num_sub_sensors, num_samples, 1, 0)), 
            'val': np.empty((num_sub_sensors, num_samples, 1, 0)), 
            'test': np.empty((num_sub_sensors, num_samples, 1, 0)),
            'remain': np.empty((num_sub_sensors, num_samples, 1, 0))}
    
    # To guarantee the same samples from multiple sensors from the same measurement 
    if 'rng_val' in load_struct:
        rng_val = load_struct['rng_val']
    else:
        rng_val = 0

    if random_flag:
        np.random.seed(None)
        rng_val = np.random.randint(1, 1000)

    

    
    for k in range(len(file_name_data_all)):
        file_name_data = file_name_data_all[k]
        file_name_target = file_name_target_all[k]
    
        print(file_name_data)
        print(target_gas)
        print(method_names[load_method])
        print("RandomFlag")
        print(random_flag)

        # Load Data
        sensor_struct = sio.loadmat(file_name_data)
        targets = sio.loadmat(file_name_target)
        sensor_cell = [sensor_struct[key] for key in sensor_struct if not key.startswith("__")]

        # Target size
        if 'numRetTar' not in load_struct:
            target_gas_t = targets[target_gas].T
        else:
            temp_target = [targets[key] for key in targets if key not in ['__header__', '__version__', '__globals__']]
            target_gas_t = np.hstack([temp_target[i] for i in load_struct['numRetTar']]).T
        
        # Bring data in expected form
        for lv_sub_sens in range(num_sub_sensors):
            if num_samples != sensor_cell[lv_sub_sens].shape[1]:
                x = np.arange(sensor_cell[lv_sub_sens].shape[1])
                y = np.linspace(1, sensor_cell[lv_sub_sens].shape[1], num_samples)
                data_t = np.zeros((sensor_cell[lv_sub_sens].shape[0], num_samples))
                for j in range(sensor_cell[0].shape[0]):
                    data_t[j, :] = np.interp(y, x, sensor_cell[lv_sub_sens][j, :])
                sensor_cell[lv_sub_sens] = data_t
        
        # Normalize data
        if norm_flag:
            for i in range(sensor_cell[0].shape[0]):  # iterate over each observation
                combined_sensors = np.vstack([sensor_cell[lv_sub_sens][i, :] for lv_sub_sens in range(num_sub_sensors)])
                scaler = StandardScaler()
                normalized_sensors = scaler.fit_transform(combined_sensors.T).T
                for lv_sub_sens in range(num_sub_sensors):
                    sensor_cell[lv_sub_sens][i, :] = normalized_sensors[lv_sub_sens, :]
        
        # Go Through Method
        if load_method == 1:
            # Method 1: 80-20 split of Unique Gas Mixtures (UGMs) for training and testing respectively
            ranges = targets['range'][~np.isnan(targets['range'])]
            u_ranges = np.unique(ranges)
            if random_flag:
                np.random.seed(rng_val)
            else:
                np.random.seed(rng_val)
            u_ranges = np.random.permutation(u_ranges)
            total_length = len(u_ranges)
            test_set = int(0.2 * len(u_ranges))
            sec_train = u_ranges[test_set:total_length]

        elif load_method == 2:
            # Method 2: Transfer learning data split
            ranges = targets['range'][~np.isnan(targets['range'])]
            u_ranges = np.unique(ranges)
            np.random.seed(rng_val)
            u_ranges = np.random.permutation(u_ranges)
            total_length = len(u_ranges)
            test_set = int(0.2 * len(u_ranges))
            if random_flag:
                np.random.seed(rng_val)
                sec_train = np.random.permutation(u_ranges[test_set:total_length])[:int(total_length * load_struct['transf'])]
            else:
                sec_train = u_ranges[test_set:total_length][:load_struct['transf']]

        elif load_method == 3:
            # Method 3: Drift or change the model over time
            ranges = targets['range'][~np.isnan(targets['range'])]
            u_ranges = np.unique(ranges)[::-1]
            total_length = len(u_ranges)
            test_set = int(load_struct['driftTest'] * len(u_ranges))
            sec_train = u_ranges[int((1-load_struct['driftTrain']) * total_length):]
            if random_flag:
                np.random.seed(rng_val)
                sec_train = np.random.permutation(sec_train)

        elif load_method == 4:
            # Method 4: Split based on measurements
            ranges = targets['range'][~np.isnan(targets['range'])].flatten()
            train_meas = load_struct['measurements'][0]
            test_meas = load_struct['measurements'][1]
            u_ranges = []
            for meas in test_meas:
                u_ranges.extend(ranges[targets['measurement'].flatten() == meas])
            u_ranges = np.unique(u_ranges)
            test_set = len(u_ranges)
            sec_train = []
            for meas in train_meas:
                sec_train.extend(ranges[targets['measurement'].flatten() == meas])
            sec_train = np.unique(sec_train)

        elif load_method == 5:
            # Method 5: Field test data split
            ranges = targets['fieldtest'][~np.isnan(targets['fieldtest'])]
            u_ranges = np.unique(ranges)
            if random_flag:
                np.random.seed(rng_val)
            else:
                np.random.seed(rng_val)
            total_length = len(u_ranges)
            test_set = int(0.99 * len(u_ranges))
            sec_train = u_ranges[test_set:total_length]
            targets['range'] = targets['fieldtest']

        elif load_method == 6:
            # Method 6: Specific UGMs based on measurements
            ranges = targets['range'][~np.isnan(targets['range'])].flatten()
            train_meas = load_struct['measurements'][0]
            test_meas = load_struct['measurements'][1]
            u_ranges = []
            for meas in test_meas:
                u_ranges.extend(ranges[ranges == meas])
            u_ranges = np.unique(u_ranges)
            test_set = len(u_ranges)
            sec_train = []
            for meas in train_meas:
                sec_train.extend(ranges[ranges == meas])
            sec_train = np.unique(sec_train)

        
        
        # Create Data Arrays
        ind = ~np.isnan(targets['range']).flatten()
        ind2 = np.zeros_like(ind, dtype=bool)
        ind3 = np.zeros_like(ind, dtype=bool)
        ind4 = np.zeros_like(ind, dtype=bool)

        for lv1 in range(len(sec_train)):
            if lv1 % 8 > 0:
                ind2[targets['range'].flatten() == sec_train[lv1]] = ind[targets['range'].flatten() == sec_train[lv1]]
                ind[targets['range'].flatten() == sec_train[lv1]] = False
            else:
                ind3[targets['range'].flatten() == sec_train[lv1]] = ind[targets['range'].flatten() == sec_train[lv1]]
                ind[targets['range'].flatten() == sec_train[lv1]] = False
        
        for lv1 in range(test_set):
            ind4[targets['range'].flatten() == u_ranges[lv1]] = ind[targets['range'].flatten() == u_ranges[lv1]]
            ind[targets['range'].flatten() == u_ranges[lv1]] = False

        ind4[targets['range'].flatten() == 1] = False
        ind4[targets['range'].flatten() == 2] = False
        ind4[targets['range'].flatten() == 3] = False
        ind3[targets['range'].flatten() == 1] = False
        ind3[targets['range'].flatten() == 2] = False
        ind3[targets['range'].flatten() == 3] = False
        ind2[targets['range'].flatten() == 1] = False
        ind2[targets['range'].flatten() == 2] = False
        ind2[targets['range'].flatten() == 3] = False
        


        remain_target = target_gas_t[ind, :]
        train_target = target_gas_t[ind2, :]
        val_target = target_gas_t[ind3, :]
        test_target = target_gas_t[ind4, :]
    
        train_data_mat3 = [sensor_cell[lv_sub_sens][ind2, :] for lv_sub_sens in range(num_sub_sensors)]
        train_data_mat4d = np.zeros((num_sub_sensors, train_data_mat3[0].shape[1], 1, train_data_mat3[0].shape[0]))
        for i in range(train_data_mat3[0].shape[0]):
            for lv_sub_sens in range(num_sub_sensors):
                train_data_mat4d[lv_sub_sens, :, 0, i] = train_data_mat3[lv_sub_sens][i, :]
    
        val_data_mat3 = [sensor_cell[lv_sub_sens][ind3, :] for lv_sub_sens in range(num_sub_sensors)]
        val_data_mat4d = np.zeros((num_sub_sensors, val_data_mat3[0].shape[1], 1, val_data_mat3[0].shape[0]))
        for i in range(val_data_mat3[0].shape[0]):
            for lv_sub_sens in range(num_sub_sensors):
                val_data_mat4d[lv_sub_sens, :, 0, i] = val_data_mat3[lv_sub_sens][i, :]
        
        test_data_mat3 = [sensor_cell[lv_sub_sens][ind4, :] for lv_sub_sens in range(num_sub_sensors)]
        test_data_mat4d = np.zeros((num_sub_sensors, test_data_mat3[0].shape[1], 1, test_data_mat3[0].shape[0]))
        for i in range(test_data_mat3[0].shape[0]):
            for lv_sub_sens in range(num_sub_sensors):
                test_data_mat4d[lv_sub_sens, :, 0, i] = test_data_mat3[lv_sub_sens][i, :]

        remain_data_mat3 = [sensor_cell[lv_sub_sens][ind, :] for lv_sub_sens in range(num_sub_sensors)]
        remain_data_mat4d = np.zeros((num_sub_sensors, remain_data_mat3[0].shape[1], 1, remain_data_mat3[0].shape[0]))
        for i in range(remain_data_mat3[0].shape[0]):
            for lv_sub_sens in range(num_sub_sensors):
                remain_data_mat4d[lv_sub_sens, :, 0, i] = remain_data_mat3[lv_sub_sens][i, :]


        target['train'] = np.concatenate((target['train'], train_target), axis=0) if target['train'].size else train_target
        target['val'] = np.concatenate((target['val'], val_target), axis=0) if target['val'].size else val_target
        target['test'] = np.concatenate((target['test'], test_target), axis=0) if target['test'].size else test_target
        target['remain'] = np.concatenate((target['remain'], remain_target), axis=0) if target['remain'].size else remain_target
    
        data['train'] = np.concatenate((data['train'], train_data_mat4d), axis=3) if data['train'].size else train_data_mat4d
        data['val'] = np.concatenate((data['val'], val_data_mat4d), axis=3) if data['val'].size else val_data_mat4d
        data['test'] = np.concatenate((data['test'], test_data_mat4d), axis=3) if data['test'].size else test_data_mat4d
        data['remain'] = np.concatenate((data['remain'], remain_data_mat4d), axis=3) if data['remain'].size else remain_data_mat4d


        targetPlot =  targets[target_gas][~np.isnan(targets['range'])]
        indC = ind[~np.isnan(targets['range'].flatten())]
        ind2C = ind2[~np.isnan(targets['range'].flatten())]
        ind3C = ind3[~np.isnan(targets['range'].flatten())]
        ind4C = ind4[~np.isnan(targets['range'].flatten())]
        overlayTarget = -1*np.ones(targetPlot.shape)
        dot_size = 5
        dot_size2 = 1
        alphaVal = 0.2
        alphaVal2 = 0.6
        # Custom marker using rectangle
        train_marker = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

        # Adjust the size of the marker
        train_marker[:, 0] *= 1
        train_marker[:, 1] *= 5
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.plot(targetPlot)
        plt.scatter(np.where(indC),targetPlot[indC], label='remaining', alpha=alphaVal2, c='g',s=dot_size)
        plt.scatter(np.where(indC),overlayTarget[indC], alpha=alphaVal, c='g',s=dot_size2, marker='s')
        plt.scatter(np.where(ind2C),targetPlot[ind2C], label='train', alpha=alphaVal2, c='b',s=dot_size)
        plt.scatter(np.where(ind2C),overlayTarget[ind2C], alpha=alphaVal, c='b',s=dot_size2, marker='s')
        plt.scatter(np.where(ind3C),targetPlot[ind3C], label='val', alpha=alphaVal2, c='r',s=dot_size)
        plt.scatter(np.where(ind3C),overlayTarget[ind3C], alpha=alphaVal, c='r',s=dot_size2, marker='s')
        plt.scatter(np.where(ind4C),targetPlot[ind4C], label='test', alpha=alphaVal2, c='y',s=dot_size)
        plt.scatter(np.where(ind4C),overlayTarget[ind4C], alpha=alphaVal, c='y',s=dot_size2, marker='s')
        plt.xlabel('Samples')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('Target over time')
        plt.grid(True)
        plt.show()

    if load_struct['saveFlag']:
        sio.savemat(load_struct['saveName'], {"data": data, "target": target})

    data["train"] = np.transpose(data["train"], (3, 0, 1, 2))
    data["val"] = np.transpose(data["val"], (3, 0, 1, 2))
    data["test"] = np.transpose(data["test"], (3, 0, 1, 2))
    data["remain"] = np.transpose(data["remain"], (3, 0, 1, 2))
    
    return data, target

