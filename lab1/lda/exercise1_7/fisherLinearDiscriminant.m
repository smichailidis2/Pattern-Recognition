function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1)'; % mean value of X1
    mu2 = mean(X2)'; % mean value of X2

    S1 = (1/m1)*(X1'*X1); % covariance matrix of X1
    S2 = (1/m2)*(X2'*X2); % covariance matrix of X2

    Sw = (S1 + S2)/2;% Within class scatter matrix
    Sb = mu1 - mu2;
    
    % optimal direction for maximum class separation 
    %v =  inv(Sw)*Sb;
    v =  (Sw)\Sb;

    v = v/norm(v) % return a vector of unit norm
