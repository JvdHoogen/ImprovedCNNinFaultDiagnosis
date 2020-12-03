# Read imports and utility functions
exec(open('import_file.py').read())
exec(open('Utils.py').read())

# Full WDCNN model function
def full_model_WDCNN():
    # load in data
    data = multivariate_cwru.CWRU("12FanEndFault",2048,0.8,1,2,'1797','1750',normal_condition = True)
    
    # Sequence length
    seq_len = data.X_train.shape[1]

    # Number of sensors
    sens = data.X_train.shape[2]

    # Create dummies for the labels
    data.y_train = pd.DataFrame(data.y_train, columns=['label'])
    dummies = pd.get_dummies(data.y_train['label']) # Classification
    products = dummies.columns
    y = dummies.values

    # Initialize a scaler using the training data.
    scaler = StandardScaler().fit(flatten(data.X_train))

    ## scaling of train,validation and test set
    data.X_train = scale(data.X_train, scaler)
    data.X_test = scale(data.X_test, scaler)

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
    
    
        # Create model
        input_shape = (seq_len,sens)

        left_input = Input(input_shape)
        convnet = Sequential()

        # WDCNN architecture
        convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same',input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation("relu"))
        convnet.add(MaxPooling1D(strides=2))
    
        convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
        convnet.add(BatchNormalization())
        convnet.add(Activation("relu"))
        convnet.add(MaxPooling1D(strides=2))
    
        convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
        convnet.add(BatchNormalization())
        convnet.add(Activation("relu"))
        convnet.add(MaxPooling1D(strides=2))
    
        convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
        convnet.add(BatchNormalization())
        convnet.add(Activation("relu"))
        convnet.add(MaxPooling1D(strides=2))
    
        convnet.add(Conv1D(filters=64, kernel_size=3, strides=1))
        convnet.add(BatchNormalization())
        convnet.add(Activation("relu"))
        convnet.add(MaxPooling1D(strides=2))
    
        convnet.add(Flatten())
        convnet.add(Dense(100,activation='sigmoid'))

        convnet.add(Dropout(0.5))
        convnet.add(Dense(data.nclasses, activation = 'softmax'))
        print(convnet.summary())


        print(convnet.count_params())

        # initialize optimizer and random generator within one fold
        keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
        keras.optimizers.SGD(lr=0.01)
        convnet.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Fit the model
        convnet.fit(x_train, y_train,validation_data = (x_test,y_test), epochs = 500, batch_size = 32, verbose=0, 
                     callbacks =[earlystop], shuffle = True)
    
        
        #Output extraction from every layer
        layer_outputs = [layer.output for layer in convnet.layers[:]] 

        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=convnet.input, outputs=layer_outputs) 
        
        # Predictions on the validation set
        predictions = convnet.predict(x_test)
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
        test_predictions_loop = convnet.predict(data.X_test)
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
        activations = activation_model.predict(data.X_test)
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



# Initialize the full_model_WDCNN function 
oos_test_y = []
oos_test_prob = []
aggregated_score = 0
aggregated_test_score = 0
earlystop = 0
runtime = 0
oos_test_activations = []
oos_test_y = []
   

oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, earlystop, oos_test_activations, oos_test_y = full_model_WDCNN()





