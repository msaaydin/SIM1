function [ImDc,Implus,ImMin,AvNoiseDc,AvNoisePlus,AvNoiseMin,modulationF]...
    = WienerFilterAndShift(FImDc,FImplus,FImin,OTFo,ObjParam,modulationFreqVector,Kotf,kConstPowAndMultConst,...
    kElementSquare)

% Shift separated comp. to exact pozition 

w = size(OTFo,1);
wo = w/2;
x = gpuArray.linspace(0,w-1,w);
y = gpuArray.linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);

% suppressing out-of-focus signal of off-center components using
% G is a notch-filter (estimated heuristically)
G = 1 - exp(-0.05*Ro.^1.2);
% dimx = 32; % tile size X
% dimy = 32;
% grid = [ceil(((w+dimx-1)/dimx)) ceil(((w+dimy-1)/dimy))];
% kProd= parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','pointProductMatrixGPU');
% kProd.ThreadBlockSize  = [dimx dimy];
% kProd.GridSize         = grid;
FImplus = FImplus.*G;
FImin = FImin.*G;



% filter central freq. with wiener filt.
SFo = 1;
co = 1.0;
[ImDc,AvNoiseDc] = CenterFreqFilterWiener(FImDc,OTFo,co,ObjParam,SFo,...
    Kotf,kConstPowAndMultConst,kElementSquare);


modulationF = DefineModFactor(FImplus,modulationFreqVector,ObjParam,OTFo,Kotf);


kv = modulationFreqVector(2) + 1i*modulationFreqVector(1);
Rplus = abs(Cv-kv);
Rmin = abs(Cv+kv);

OBJplus = zeros(w,'gpuArray');
OBJmin = zeros(w,'gpuArray');
OBJplus = feval(kConstPowAndMultConst,Rplus,ObjParam(2),ObjParam(1),OBJplus,w,w); % Aobj*(Ro.^Bobj) running on GPU
OBJmin = feval(kConstPowAndMultConst,Rmin,ObjParam(2),ObjParam(1),OBJmin,w,w); % Aobj*(Ro.^Bobj) running on GPU



k3 = round(modulationFreqVector);
OBJplus(wo+1+k3(1),wo+1+k3(2)) = 0.25*OBJplus(wo+2+k3(1),wo+1+k3(2))...
	+ 0.25*OBJplus(wo+1+k3(1),wo+2+k3(2))...
	+ 0.25*OBJplus(wo+0+k3(1),wo+1+k3(2))...
	+ 0.25*OBJplus(wo+1+k3(1),wo+0+k3(2));
OBJmin(wo+1-k3(1),wo+1-k3(2)) = 0.25*OBJmin(wo+2-k3(1),wo+1-k3(2))...
	+ 0.25*OBJmin(wo+1-k3(1),wo+2-k3(2))...
	+ 0.25*OBJmin(wo+0-k3(1),wo+1-k3(2))...
	+ 0.25*OBJmin(wo+1-k3(1),wo+0-k3(2));

% side lobes Filtering 

[fDpf,AvNoisePlus] = SideFreqFilterWiener(FImplus,OTFo,co,OBJmin,modulationF,Kotf);
[fDmf,AvNoiseMin] = SideFreqFilterWiener(FImin,OTFo,co,OBJplus,modulationF,Kotf);

    t = w;
    to = t/2;
    u = gpuArray(linspace(0,t-1,t));
    v = gpuArray(linspace(0,t-1,t));
    [U,V] = meshgrid(u,v);

% Shift separated freq comp. to correct position.
Implus = fft2(ifft2(fDpf).*exp( +1i.*2*pi*(modulationFreqVector(2)/t.*(U-to) + modulationFreqVector(1)/t.*(V-to)) ));
ImMin = fft2(ifft2(fDmf).*exp( -1i.*2*pi*(modulationFreqVector(2)/t.*(U-to) + modulationFreqVector(1)/t.*(V-to)) ));



