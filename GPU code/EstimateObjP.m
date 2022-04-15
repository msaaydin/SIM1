function PowerParam = EstimateObjP(NoisyIm,OTFo,cutoff,kConstPowAndMultConst,...
    CuMatCompareG,CuMatCompareL)
% object power param. determination


OTFo = gpuArray(OTFo);
w = size(OTFo,1);
wo = w/2;
x = gpuArray(linspace(0,w-1,w));
y = gpuArray(linspace(0,w-1,w));
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);
Ro(wo+1,wo+1) = 1; % for nan

% OTF cut-off frequency = cutoff;
Zm = (Ro>0.3*cutoff).*(Ro<0.4*cutoff);
Aobj = gather(sum(sum(abs(NoisyIm.*Zm)))./sum(sum(Zm)));
Bobj = -0.5;
ZeroObj = [Aobj Bobj];


[nx,ny] = size(NoisyIm);
dimx = 32; % tile size X
dimy = 32;
grid = [ceil(((nx+dimx-1)/dimx)) ceil(((ny+dimy-1)/dimy))];
kernelSub = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','subtractMatrixGPU');
kernelSub.ThreadBlockSize  = [dimx dimy];
kernelSub.GridSize         = grid;

kProd= parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','pointProductMatrixGPU');
kProd.ThreadBlockSize  = [dimx dimy];
kProd.GridSize         = grid;


OBJparaOpt0 = @(ZeroObj1)PowerObjOpt(ZeroObj1,NoisyIm,OTFo,cutoff,Ro);
options = optimset('LargeScale','off','Algorithm',...
	'active-set','MaxFunEvals',50,'MaxIter',50,'Display','notify');

% optimize
[PowerParam] = fminsearch(OBJparaOpt0,ZeroObj,options);

