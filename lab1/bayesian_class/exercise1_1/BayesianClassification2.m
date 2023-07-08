%% PATTERN REVOGNITION AND STATISTICAL MODELING %%

% Michailidis Stergios 2020030080
% Moustakas   Ioannis  2020030120

% Exercise 1, question (1) , d
close all;
clear
clc

miu1   = [3 3];
sigma1 = [1.2  0.4 ; 0.4  1.2];

miu2   = [6 6];
sigma2 = sigma1

x1 = 0 : 0.25 : 10 - 0.25;
x2 = 0 : 0.25 : 10 - 0.25;

[X1,X2] = meshgrid(x1,x2);

G1 = mvnpdf([X1(:) X2(:)], miu1, sigma1);
%reshape to match matrix dimension of X1
G1 = reshape(G1, size(X1));

G2 = mvnpdf([X1(:) X2(:)], miu2, sigma2);
%reshape to match matrix dimension of X2 
G2 = reshape(G2, size(X2));

figure(1)
hold on
grid on
contour(X1,X2,G1, 30);
contour(X1,X2,G2, 30);
hold off
xlabel('X 1')
ylabel('X 2')

figure(2)
hold on
grid on
surf(X1,X2,G1);
surf(X1,X2,G2);
hold off

% a-priori probabilities
apriori1 = [0.1 0.25 0.5 0.75 0.9];
apriori2 = 1 - apriori1;
k = apriori2./apriori1;

% N number of samples
N = 300;

data_x1 = mvnrnd(miu1,sigma1,N);
data_x2 = mvnrnd(miu2,sigma2,N);

x_2 = -1:0.1:10;

% decision boundary x_1 = 9 - 0.533*log(k) - x_2
for i = k
    figure
    plot(data_x1(:,1),data_x1(:,2),'r.',data_x2(:,1),data_x2(:,2),'k.')
    hold on
    x_1 = 9 - 0.533*log(i) - x_2
    plot(x_2,x_1,"LineWidth",2)
    hold off
    axis([-1 10 -1 10]);
end
