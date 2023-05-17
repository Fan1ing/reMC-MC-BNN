
 clc
clear all
data = load('C:\Users\Arthas\Desktop\train of reMC-ANN.csv');%Training data
data1 = load('C:\Users\Arthas\Desktop\test of reMC-ANN.csv');%test data
i=1;
T_sim1 =0;
while i<=10
     P_train = data(:,1:86);
     T_train = data(:,87:129);
     P_test = data1(:,1:86);
     T_test = data1(:,87:129);
     P_train = P_train'
     T_train = T_train'
     P_test =  P_test'
     T_test = T_test'
     [p_train , ps_train ] = mapminmax(P_train,0,1);
     p_test = mapminmax('apply',P_test,ps_train);
    [t_train , ps_output ] = mapminmax(T_train , 0,1);
    net = feedforwardnet([5],'trainbr');
    net.trainParam.epochs = 30;
    net.trainParam.goal = 0.0035;
    net.trainParam.mu = 0.008;
    net = train(net,p_train,t_train);
    t_sim = sim(net,p_test);
    T_sim = mapminmax('reverse',t_sim,ps_output);
    T_sim1= T_sim1+T_sim;
    i=i+1;
end
T_sim1=T_sim1/(i-1);