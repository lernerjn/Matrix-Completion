function [ X, rankX, M, Q ] = Min_Rank_Matrix_Completion(n1, n2, m,  r , epsilon )
% Author: Jeremy Lerner, Stony Brook University
% MinRankCompl estimates the minimum rank of an incomplete matrix, M, 
% which is sampled (randomly) from Q, the matrix we are trying to recover, 
% this is an NP hard problem (given M(i,j) and Omega and optimizing over X)
% *Note that this program requires the convex optimization library CVX*
%
%   min         rank(X)
%   subject to  X(i,j) = M(i,j) for all (i,j) in Omega
%
% The most important output is X, which is the optimal argument for the
% following convex relaxation of the above NP hard problem 
%(where ||X||_* is the nuclear norm, also called the trace norm and 
% the Ky-Fan n-norm)
%
%   min         ||X||_*
%   subect to   X(i,j) = M(i,j) for all (i,j) in Omega
%
% Furthermore, this problem is equivalent to the semidefinite program
%
%   min         (trace(Y) + trace(Z))/2
%   subject to  [ Y   X ] 
%               [ X^T Z ]   >= 0
%               X(i,j) = M(i,j) for all (i,j) in Omega
%
% This is the problem that this program actually solves, using CVX. The
% optimizer, X, is the estimate of the minimizer of the minimum rank matrix
% completion problem. The rank of X is estimated by counting the number of
% singular values, in the singular value decomposition of X, that are
% greater than epsilon.
%
% Inputs:
%       n1: number of rows in M, the matrix we are trying to complete
%       n2: number of columns in M, the matrix we are trying to complete
%           (i.e. M is n1 by n2)
%       m:  the number of observations, that is, the number of elements of
%           Q that will be randomly selected to compose M
%       r:  the rank of Q
%       epsilon: the tolerance for counting singular values of X
%
% Optional Alternative Input (two inputs exactly):
%       M: the matrix to be completed, all numbers are used and all NaN
%       entries are ignored
%       epsilon: the tolerance for counting singular values of X
%
% Outputs:
%       X: the minimizer of this convex relaxation of the minimum rank 
%           matrix completion problem
%       rankX: the rank of X, estimated by counting the number of singular
%           values >= epsilon
%       M: the matrix of observations
%       Q: the matrix we are trying to recover

%% The setup

if ~(nargin() == 2)
% Create an n1 x n2 matrix of rank r
    Q = randn(n1,r)*randn(r,n2);

    % Q = [ 1 1 1; 2 2 2; 3 3 3];

    % Randomly (from a uniform distribution) pick m elements in Q to place in M
    list1=randperm(numel(Q));
    list1=list1(1:m);
    M = nan(n1,n2);
    M(list1) = Q(list1);

    % M = [ nan nan 1; 2 2 nan ; 3 3 3];
else
    M = n1;
    epsilon = n2;
end


%% The Convex Optimization

[m,n] = size(M);
% Find the locations where X(i,j) = M(i,j), all numbers are in Omega and 
% all NaN entries are not
nan_index = find(~isnan(M));

% Solve the convex optimization problem in CVX
cvx_begin
    variables X(m,n) Y(m,m) Z(n,n)
    minimize trace(Y)/2 + trace(Z)/2
    subject to
        [Y X; X' Z] == semidefinite(n+m);
        X(nan_index) == M(nan_index);
        
cvx_end

% Calculate the singular value decomposition of X
[~, S, ~] = svd(full(X)); 
% The number of singular values >= epsilon is the rank of X
rankX = numel(find(diag(S)>epsilon));

end