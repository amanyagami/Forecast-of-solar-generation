%course_project
%carefully selected the clearday data plots 
a=table2array(Generationdata(76:459,2));
b=table2array(Irradiationdata(97:480,2));
TestIp=a;
TrainIp=b;



                                             
TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                     % remove noise (less than 0) from DATA
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title("  clear day Unit Generation 15 min")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network
oo=YPred-YTest;
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));
XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast clear day 15 min")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast clear day 15 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er +" MAPE = "+mape_er)

%course_project
%carefully selected the clearday data plots 
a1=table2array(Generationdata(76:459,2));
b1=table2array(Irradiationdata(97:480,2));
TestIp=a1;
TrainIp=b1;
size(TrainIp)

TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                     % remove noise (less than 0) from DATA
                                                    
a=[];
for g =1:192
    a=[a;TestIp(g*2)];
end
b=[];
for g =1:192
    b=[b;TrainIp(g*2)];
end
TestIp=a;
TrainIp=b;
size(TrainIp)
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    


mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);    
size(input)

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title("  clear day Unit Generation 30 min")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network
oo=YPred-YTest;
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast clear day 30 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast clear day 30 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er +" MAPE = "+mape_er)

%course_project
%carefully selected the clearday data plots 
a=table2array(Generationdata(76:459,2));
b=table2array(Irradiationdata(97:480,2));
TestIp=a;
TrainIp=b;

TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                    % remove noise (less than 0) from DATA
a=[];
for g =1:128
    a=[a;TestIp(g*3)];
end
b=[];
for g =1:128
    b=[b;TrainIp(g*3)];
end
TestIp=a;
TrainIp=b;
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title("  clear day Unit Generation 45")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network
oo=YPred-YTest;

a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast clear day 45 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast clear day 45 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er +" MAPE = "+mape_er)

%course_project
%carefully selected the clearday data plots 
a=table2array(Generationdata(76:459,2));
b=table2array(Irradiationdata(97:480,2));
TestIp=a;
TrainIp=b;

TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                   % remove noise (less than 0) from DATA
a=[];
for g =1:96
    a=[a;TestIp(g*4)];
end
b=[];
for g =1:96
    b=[b;TrainIp(g*4)];
end
TestIp=a;
TrainIp=b;
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title("  clear day Unit Generation 60 min")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network
oo=YPred-YTest;

a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast clear day 60 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast clear day 60 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er +" MAPE = "+mape_er)

%carefully selected the cloudy data plots 
a=table2array(Generationdata(1420:1611,2));

a1=table2array(Generationdata(1804:1995,2));


b=table2array(Irradiationdata(1441:1632,2));

b1=table2array(Irradiationdata(1825:2016,2));


TestIp=[a;a1];
TrainIp=[b;b1];




TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;   
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                     % remove noise (less than 0) from DATA

TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    
s=size(TrainIp)
s1=size(TestIp)

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title(" Unit Generation in cloudy day 15 mins")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network

oo=YPred-YTest;

a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("day ")
ylabel("Cases")
title("Forecast in cloudy day 15 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("generation")
title("Forecast in cloudy day 15 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse = sqrt(mean((YPred-YTest).^2))                      

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("generation")
xlabel("time ")
title("Forecast with Updates in cloudy day  15 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

%carefully selected the cloudy data plots 
a=table2array(Generationdata(1420:1611,2));

a1=table2array(Generationdata(1804:1995,2));


b=table2array(Irradiationdata(1441:1632,2));

b1=table2array(Irradiationdata(1825:2016,2));


TestIp=[a;a1];
TrainIp=[b;b1];




TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                         % remove noise (more than 50) from DATA

TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                     % remove noise (less than 0) from DATA
a=[];
for g =1:192
    a=[a;TestIp(g*2)];
end
b=[];
for g =1:192
    b=[b;TrainIp(g*2)];
end
TestIp=a;
TrainIp=b;
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    
s=size(TrainIp)
s1=size(TestIp)

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title(" Unit Generation in cloudy day 30 mins")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network

oo=YPred-YTest;
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("day ")
ylabel("Cases")
title("Forecast in cloudy day 30 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("generation")
title("Forecast in cloudy day 30 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse = sqrt(mean((YPred-YTest).^2))                      

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("generation")
xlabel("time ")
title("Forecast with Updates in cloudy day 30 mins ")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

%carefully selected the cloudy data plots 
a=table2array(Generationdata(1420:1611,2));

a1=table2array(Generationdata(1804:1995,2));


b=table2array(Irradiationdata(1441:1632,2));

b1=table2array(Irradiationdata(1825:2016,2));


TestIp=[a;a1];
TrainIp=[b;b1];




TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                        % remove noise (less than 0) from DATA
a=[];
for g =1:128
    a=[a;TestIp(g*3)];
end
b=[];
for g =1:128
    b=[b;TrainIp(g*3)];
end
TestIp=a;
TrainIp=b;
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    


mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title(" Unit Generation in cloudy day 45 mins")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network

oo=YPred-YTest;
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("day ")
ylabel("Cases")
title("Forecast in cloudy day 45 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("generation")
title("Forecast in cloudy day 45 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse = sqrt(mean((YPred-YTest).^2))                      

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("generation")
xlabel("time ")
title("Forecast with Updates in cloudy day 45 mins ")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

%carefully selected the cloudy data plots 
a=table2array(Generationdata(1420:1611,2));

a1=table2array(Generationdata(1804:1995,2));


b=table2array(Irradiationdata(1441:1632,2));

b1=table2array(Irradiationdata(1825:2016,2));


TestIp=[a;a1];
TrainIp=[b;b1];




TrainIp(isnan(TrainIp)) =0;                                               % remove NAN from DATA                                             % remove noise (more than 50) from DATA
TestIp(TestIp>50)=30;                                                       % remove noise (more than 50) from DATA                                                  % remove noise (less than 0) from DATA
TestIp(TestIp<0)=0;                                                        % remove noise (less than 0) from DATA                                                     % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=0;                                                       % remove noise (less than 0) from DATA
a=[];
for g =1:96
    a=[a;TestIp(g*4)];
end
b=[];
for g =1:96
    b=[b;TrainIp(g*4)];
end
TestIp=a;
TrainIp=b;
TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';    
s=size(TrainIp)
s1=size(TestIp)

mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);             

figure
plot(input)
hold on
plot(target,'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title(" Unit Generation in cloudy day 60 mins")

numTimeStepsTrain = floor(0.75*numel(input));                               % 75 and 25 percent training and testing points

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);          

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training

[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse = sqrt(mean((YPred-YTest).^2))                                         % error of network

oo=YPred-YTest;
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));

XTestIp = (mx2-mn2)*XTestIp + mn2;                      

figure
plot(XTestIp(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("day ")
ylabel("Cases")
title("Forecast in cloudy day 60 mins")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("generation")
title("Forecast in cloudy day 60 mins")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse = sqrt(mean((YPred-YTest).^2))  
oo=YPred-YTest
a=YTest(YTest>0);
b=YPred(YTest>0);
mae_er=mae(oo);
mape_er=mean(abs((b-a)/a));


figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("generation")
xlabel("time ")
title("Forecast with Updates in cloudy day 60 day ")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse+" MAE = "+mae_er+" MAPE = "+mape_er)

zip('backup',{'*.m','*.mlx'});

