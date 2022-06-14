function [affinitymat_0,I,I_nozero] = top(affinitymat_0)
% return top k of affinitymat, return displaced index, return nozero index 
[row, line] = size(affinitymat_0);
k = 50
I = zeros(row,line);
I_nozero = zeros(row,k);
for i = 1:row
    tmp0 = affinitymat_0(i,:);
    [tmp1,I(i,:)] = sort(tmp0,'descend');
    tmp0(tmp0<tmp1(k)) = 0;
    affinitymat_0(i,:) = tmp0 / sum(tmp0);
%     I_nozero(i,:) = find(affinitymat_0(i,:)~=0);
end
end