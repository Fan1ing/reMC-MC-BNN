
 clc
clear all
data = load('C:\Users\Arthas\Desktop\DATASETS.csv');%Training data
data1 = load('C:\Users\Arthas\Desktop\FIND.csv');%test data
i=1;
mae= 10000000000000000000000000;
ys = 0;% Most similar spectral vector
while i<=71%Total number of data
     x = data(i,:);
     y = data1(1,:);
    mae1 = mean(abs(x-y));
    if mae1<mae
        mae = mae1
        ys = x
    else
        mae=mae
        ys=ys
    end
    i=i+1;
end
fprintf ('MAE=%d/', mae);