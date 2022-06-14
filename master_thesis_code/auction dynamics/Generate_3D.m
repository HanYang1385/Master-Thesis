N = 40;
step = [10,20,20,50,80];
%step = 100;
Step_end = length(step);
p = 4;
q = zeros(1,p);
q(1) = 14;
q(2) = 23;
q(3) = 30;
q(4) = 33;
L = sum(q);
x = 1:N;
y = x;
z = x;
[X1,X2,X3] = meshgrid(x,y,z);
X1 = X1(:);
X2 = X2(:);
X3 = X3(:);
Y = (1:N)'*ones(1,N);
v = (X1/N - 0.5).^2 + (X2/N - 0.5).^2 + (X3/N - 0.5).^2;

r = 0.06;
v(v > r) = 1;
v(v <= r) = 0;
X = Y';
%U = zeros(N);
%U = X + (Y-1) * p;
%U = floor((X - 1) / N * p1) + p1 * floor((Y - 1) / N * p1);
%U_1 = U(:);
%U = floor((X - 1) / N * p);
U = unidrnd(p + L,[N,N,N]);
% V = zeros([N,N,N]);
low = 0;
high = 0;
for i = 1:p
    low = high + 1;
    high = low + q(i);
    U((low <= U) &(U <= high)) = i;
end
U = U - 1;
U = U(:);
U(v == 1) = p-1;
% close all






% h = figure;
figure(11)
% scatter3(X1(v == 0),X2(v == 0),X3(v == 0),60,U(v == 0),'filled');
scatter3(X1(U<3),X2(U<3),X3(U<3),60,U(U<3),'filled');
colormap(turbo);
%shading interp;
shading flat;          
colorbar


% close all
% contourf(U,0:p-1,'LineStyle','none');
% colormap(jet);

% sigma = rand(p);
% sigma = sigma + sigma';
% sigma = 4 + sigma;
% % sigma = ones(p);
% fid = fopen( 'Phase_coefficient.txt' , 'w');
% for j = 1:p
%     fprintf(fid,'%f ',sigma(j,:)); 
%     fprintf(fid,'\n'); 
% end
% fclose(fid);

fid = fopen( 'Data.txt' , 'w');
fprintf(fid,'%d \n',U); 
fclose(fid);

fid = fopen( 'Output.txt' , 'r');
A = fscanf(fid,'%d');
fclose(fid);
W = U;





loops = 150;
M(loops) = struct('cdata',[],'colormap',[]);
for i = 1:20
    fid = fopen( 'Data.txt' , 'w');
    fprintf(fid,'%d \n',W); 
    fclose(fid);
    
    fid = fopen( 'Settings.txt' , 'w');
    fprintf(fid,'dim: %d, phases number: %d, points: %d, wide: %f\n',3,p,N,1/N);
    fprintf(fid,'eps_0: %f, rate: %f, eps_min: %f, iteration step: %d, delta_t: %f', 0.002, 2, 0.0004, i, 0.0001);
    fclose(fid);
    % dim: 2, phases number: 4, points: 100, wide: 0.01
    % eps_0: 0.1, rate: 2.0, eps_min: 0.01, iteration step: 27, delta_t: 0.002

    fprintf('Please make it run until it close.\n');
    testAC;
    fid = fopen( 'Output.txt' , 'r');
    A = fscanf(fid,'%d');
    fclose(fid);

    W = A;
    figure(12)
    scatter3(X1(A < 3),X2(A < 3),X3(A < 3),60,A(A < 3),'filled');
    colormap(turbo);
    shading flat;
    colorbar

    drawnow
    M(i) = getframe;
end
