
 clc
clear all
data = load('C:\Users\Arthas\Desktop\train.csv');
data1 = load('C:\Users\Arthas\Desktop\pre.csv');
P_train = data(:,1:1);
T_train = data(:,2:end);
P_test = data1(:,1:1);
T_test = data1(:,2:end);
P_train = P_train'
T_train = T_train'
P_test =  P_test'
T_test = T_test'
[p_train , ps_train ] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_train);
[t_train , ps_output ] = mapminmax(T_train , 0,1);
net = feedforwardnet([6],'trainbr');
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.00001;
net.trainParam.mu = 0.008;
net = train(net,p_train,t_train);
t_sim = sim(net,p_test);
T_sim = mapminmax('reverse',t_sim,ps_output);