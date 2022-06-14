aff = affinitymat - diag(diag(affinitymat));
% one can change top k in top.m
% one can replace ... by filename
[aff_50row,~,~] = top(aff);
W = (aff_50row + aff_50row') / 2;
D = diag(W * ones(size(aff,1),1));
P2 = 1 ./ D * W;
triuW = triu(W);
triusP = sqrt(D^(-1)) * triuW * sqrt(D^(-1));
sP = sqrt(D^(-1)) * W * sqrt(D^(-1));
[eigvec,eigval] = eig(sP);
eigvalues = diag(eigval);


ss_eigvalues = sort(eigvalues,'descend','ComparisonMethod','abs');
sr_eigvalues = sort(eigvalues,'descend','ComparisonMethod','real');
figure
plot(abs(ss_eigvalues(1:20)),'-o')
hold on
plot(sr_eigvalues(1:20),'-+')
legend('Sort by abs','Sort by real part')
title('Top 50 affinity matrix of ...')
