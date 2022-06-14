%% locate ligands and receptors pairs in TPM 
% one can replace ... by filename 
% prepreconditioning of LR_pairs to get ligands, receptors and scores,
% prepreconditioning of TPM to get genes and cells 
file1 = fopen('.../LR_pairs.txt');
genepairs = textscan(file1,'%s%s%s');
fclose(file1);
ligands = genepairs{1, 1};
receptors = genepairs{1, 2};
scores = str2double(genepairs{1, 3});

data = importdata('.../TPM.txt');
TPM = data.data;
genes = data.textdata(1:end, 1);
TPM_cell_id0 = data.textdata(1,1:end)';
% TPM_cell_id = TPM_cell_id0(2:end);

[~, ligandindex] = ismember(ligands, genes);
[~, receptorindex] = ismember(receptors, genes);
found = logical((ligandindex~=0) .* (receptorindex~=0));
ligandindex = ligandindex(found);
receptorindex = receptorindex(found);
ligands = ligands(found);
receptors = receptors(found);
scores = scores(found);
% clear data;


%% calculate affinity matrix, get affinitymat.
Atotake = ligandindex;
Btotake = receptorindex;
allscores = scores;
for i = 1:size(ligandindex,1)
    if ligandindex(i) ~= receptorindex(i)
        Atotake = [Atotake; receptorindex(i)];  %A*B + B*A
        Btotake = [Btotake; ligandindex(i)];
        allscores = [allscores; scores(i)];
    end
end
A = TPM(Atotake, :);
B = TPM(Btotake, :);
affinitymat = (diag(allscores) * A)' * B; 


%%  Identify the position number of N,P,T cells, get ind. 

file2 = fopen('.../label.txt');
labeldata = textscan(file2,'%s%s');
fclose(file2);
label_cell_id = labeldata{1,1};
label_cell_type = labeldata{1,2};

[~, cell_in_label_index0] = ismember(TPM_cell_id0,label_cell_id);
cell_in_label_index = cell_in_label_index0(2:end);
cell_in_label_type = label_cell_type(cell_in_label_index);

unique_cell_in_label_type = unique(cell_in_label_type);
len = length(unique_cell_in_label_type);
ind = cell(len,1);
i = 1;
while(i <= len)
    ind{i} = find(strcmp(cell_in_label_type,unique_cell_in_label_type{i}));
    i = i + 1;
end


%% calculate affinity between N,P,T clusters, get affinity matrix between clusters--sigma.
sigma = zeros(len,len);
for i = 1:len
    for j = 1:len
        sigma(i,j) = mean(mean(affinitymat(ind{i},ind{j})));
    end
end
sigma = sigma - diag(diag(sigma));


%% calculate the fractions of N,P,T cells, get fractions. 
fractions = zeros(len,1);
for i = 1:len
    fractions(i) = length(ind{i,1}) / size(TPM,2);
end
