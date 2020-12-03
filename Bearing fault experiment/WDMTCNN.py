# Read imports and utility functions
exec(open('import_file.py').read())
exec(open('Utils.py').read())

# Full WDMTCNN model function
def full_model_WDMTCNN():
    data = cwru.CWRU("12FanEndFault", 2048, 0.8, 1, '1797','1750')

    # Create dummies for the labels
    data.y_train = pd.DataFrame(data.y_train, columns=['label'])
    dummies = pd.get_dummies(data.y_train['label']) # Classification
    products = dummies.columns
    y = dummies.values

    # Initialize a scaler using the training data
    scaler = StandardScaler().fit(flatten(data.X_train))

    # scaling of train,validation and test set
    data.X_train = scale(data.X_train, scaler)
    data.X_test = scale(data.X_test, scaler)
    
    # Split multivariate signals into separate time series
    data.X_test0 = np.dsplit(data.X_test,2)
    data.X_test1 = data.X_test0[0]
    data.X_test2 = data.X_test0[1]
    
    # Initialize k-folds
    kf = StratifiedKFold(5, shuffle=False, random_state=42) # Use for StratifiedKFold classification
    fold = 0

    # Build empty lists for results
    oos_y = []
    oos_pred = []
    oos_test_pred = []
    oos_test_y = []
    oos_test_prob = []
    oos_test_activations = []
    
    # Earlystopping callback
    earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=30, verbose=0, mode='auto')
    
    # Initialize loop for every kth fold
    for train, test in kf.split(data.X_train, data.y_train['label']): # Must specify y StratifiedKFold for 
        fold+=1
        print(f"Fold #{fold}")
        
        x_train = data.X_train[train]
        y_train = y[train]
        x_test = data.X_train[test]
        y_test = y[test]
        
        x_train0 = np.dsplit(x_train, 2)
        x_test0 = np.dsplit(x_test,2)
        
        x_train1 = x_train0[0]
        x_train2 = x_train0[1]
        x_test1 = x_test0[0]
        x_test2 = x_test0[1]
    

        # Create multi-channel model

        # WDMTCNN channel 1
        seq_len1 = x_train1.shape[1]
        sens1 = x_train1.shape[2]
        
        # Create input shape based on sequence length
        input_shape1 = (seq_len1, sens1)
        left_input1 = Input(input_shape1)
        
        # Block 1
        conv1 = Conv1D(filters=16, kernel_size=64, strides=16, padding='same')(left_input1)
        batch1 = BatchNormalization()(conv1)
        activation1 = Activation("relu")(batch1)
        pool1 = AveragePooling1D(strides=2)(activation1)
        
        # Block 2
        conv2 = Conv1D(filters=32, kernel_size=3, strides =1, padding='same')(pool1)
        batch2 = BatchNormalization()(conv2)
        activation2 = Activation("relu")(batch2)
        pool2 = AveragePooling1D(strides=2)(activation2)
        
        # Block 3
        conv3 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool2)
        batch3 = BatchNormalization()(conv3)
        activation3 = Activation("relu")(batch3)
        pool3 = AveragePooling1D(strides=2)(activation3)
        
        # Block 4
        conv4 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool3)
        batch4 = BatchNormalization()(conv4)
        activation4 = Activation("relu")(batch4)
        pool4 = AveragePooling1D(strides=2)(activation4)        
        
        # Block 5
        conv5 = Conv1D(filters=64, kernel_size=3, strides =1)(pool4)
        batch5 = BatchNormalization()(conv5)
        activation5 = Activation("relu")(batch5)
        pool5 = AveragePooling1D(strides=2)(activation5)
        flatten1 = Flatten()(pool5)
        
        
        # WDMTCNN channel 2
        seq_len2 = x_train2.shape[1]
        sens2 = x_train2.shape[2]
        
        # Create input shape based on sequence length
        input_shape2 = (seq_len2, sens2)
        left_input2 = Input(input_shape2)
        
        # Block 6
        conv6 = Conv1D(filters=16, kernel_size=64, strides=16, padding='same')(left_input2)
        batch6 = BatchNormalization()(conv6)
        activation6 = Activation("relu")(batch6)
        pool6 = AveragePooling1D(strides=2)(activation6)
        
        # Block 7
        conv7 = Conv1D(filters=32, kernel_size=3, strides =1, padding='same')(pool6)
        batch7 = BatchNormalization()(conv7)
        activation7 = Activation("relu")(batch7)
        pool7 = AveragePooling1D(strides=2)(activation7)
        
        # Block 8
        conv8 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool7)
        batch8 = BatchNormalization()(conv8)
        activation8 = Activation("relu")(batch8)
        pool8 = AveragePooling1D(strides=2)(activation8)
        
        # Block 9
        conv9 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool8)
        batch9 = BatchNormalization()(conv9)
        activation9 = Activation("relu")(batch9)
        pool9 = AveragePooling1D(strides=2)(activation9)        
        
        # Block 10
        conv10 = Conv1D(filters=64, kernel_size=3, strides =1)(pool9)
        batch10 = BatchNormalization()(conv10)
        activation10 = Activation("relu")(batch10)
        pool10 = AveragePooling1D(strides=2)(activation10)
        flatten2 = Flatten()(pool10)        
        
        # Merge nets
        merged = concatenate([flatten1, flatten2])
        
        # Add final fully connected layers
        dense1 = Dense(100,activation='sigmoid')(merged)
        dropout = Dropout(0.5)(dense1)
        output = Dense(data.nclasses, activation = "sigmoid")(dropout)
        
        # Create combined model
        wdcnn_multi = Model(inputs=[left_input1, left_input2],outputs=output)
        print(wdcnn_multi.summary())
        
        print(wdcnn_multi.count_params())

        # initialize optimizer and random generator within one fold
        keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
        keras.optimizers.SGD(lr=0.01)
        wdcnn_multi.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        # Fit the model
        wdcnn_multi.fit([x_train1,x_train2], y_train,validation_data = ([x_test1,x_test2],y_test), epochs = 500, batch_size = 32, verbose=1, 
                     callbacks =[earlystop], shuffle = True)
    
        
        #Output extraction from every layer
        layer_outputs = [layer.output for layer in wdcnn_multi.layers[2:]] 
        
        # Creates a model that will return these outputs, given the model input
        activation_model = Model(inputs=[left_input1,left_input2], outputs=layer_outputs) 
        
        # Predictions on the validation set
        predictions = wdcnn_multi.predict([x_test1,x_test2])
        # Append actual labels of the validation set to empty list
        oos_y.append(y_test)
        # Raw probabilities to chosen class (highest probability)
        predictions = np.argmax(predictions,axis=1)
        # Append predictions of the validation set to empty list
        oos_pred.append(predictions)  
        
        # Measure this fold's accuracy on validation set compared to actual labels
        y_compare = np.argmax(y_test,axis=1) 
        score = metrics.accuracy_score(y_compare, predictions)
        print(f"Validation fold score(accuracy): {score}")
    	
    	# Predictions on the test set
        test_predictions_loop = wdcnn_multi.predict([data.X_test1,data.X_test2])
        # Append actual labels of the test set to empty list
        oos_test_y.append(data.y_test)
        # Append raw probabilities of the test set to empty list
        oos_test_prob.append(test_predictions_loop)
        # Raw probabilities to chosen class (highest probability)
        test_predictions_loop = np.argmax(test_predictions_loop, axis=1)
        # Append predictions of the test set to empty list
        oos_test_pred.append(test_predictions_loop)
        
        # Measure this fold's accuracy on test set compared to actual labels
        test_score = metrics.accuracy_score(data.y_test, test_predictions_loop)
        print(f"Test fold score (accuracy): {test_score}")
        
        # Activations per layer when predicting on test set
        activations = activation_model.predict([data.X_test1, data.X_test2])
        oos_test_activations.append(activations)
        

    # Build the prediction list across all folds
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    oos_y_compare = np.argmax(oos_y,axis=1) 

    # Measure aggregated accuracy across all folds on the validation set
    aggregated_score = metrics.accuracy_score(oos_y_compare, oos_pred)
    print(f"Aggregated validation score (accuracy): {aggregated_score}")    
    
    # Build the prediction list across all folds
    oos_test_y = np.concatenate(oos_test_y)
    oos_test_pred = np.concatenate(oos_test_pred)
    oos_test_prob = np.concatenate(oos_test_prob)
    
    # Measure aggregated accuracy across all folds on the test set
    aggregated_test_score = metrics.accuracy_score(oos_test_y, oos_test_pred)
    print(f"Aggregated test score (accuracy): {aggregated_test_score}")
    
    return(oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, earlystop.patience, 
           oos_test_activations, oos_test_y)

# Initialize the full_model_WDMTCNN function
oos_test_y = []
oos_test_prob = []
aggregated_score = 0
aggregated_test_score = 0
earlystop = 0
runtime = 0
oos_test_activations = []
oos_test_y = []
   

oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, earlystop, oos_test_activations, oos_test_y = full_model_WDMTCNN()


