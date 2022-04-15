function [ImDc,Implus,ImMin,AvNoiseDc,AvNoisePlus,AvNoiseMin,modulationF]...
    = WienerFilterAndShift(FImDc,FImplus,FImin,OTFo,ObjParam,modulationFreqVector,Kotf)

% Shift separated comp. to exact pozition 

w = size(OTFo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);

% suppressing out-of-focus signal of off-center components using
% G is a notch-filter (estimated heuristically)
G = 1 - exp(-0.05*Ro.^1.2);

FImplus = FImplus.*G;
FImin = FImin.*G;



% filter central freq. with wiener filt.
SFo = 1;
co = 1.0;
[ImDc,AvNoiseDc] = CenterFreqFilterWiener(FImDc,OTFo,co,ObjParam,SFo,Kotf);


modulationF = DefineModFactor(FImplus,modulationFreqVector,ObjParam,OTFo,Kotf);


kv = modulationFreqVector(2) + 1i*modulationFreqVector(1);
Rplus = abs(Cv-kv);
Rmin = abs(Cv+kv);
OBJplus = ObjParam(1)*(Rplus.^ObjParam(2));
OBJmin = ObjParam(1)*(Rmin.^ObjParam(2));



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
[fDmf,AvNoiseMin]  = SideFreqFilterWiener(FImin,OTFo,co,OBJplus,modulationF,Kotf);

    t = w;
    to = t/2;
    u = linspace(0,t-1,t);
    v = linspace(0,t-1,t);
    [U,V] = meshgrid(u,v);

% Shift separated freq comp. to correct position.
Implus = fft2(ifft2(fDpf).*exp( +1i.*2*pi*(modulationFreqVector(2)/t.*(U-to) + modulationFreqVector(1)/t.*(V-to)) ));
ImMin = fft2(ifft2(fDmf).*exp( -1i.*2*pi*(modulationFreqVector(2)/t.*(U-to) + modulationFreqVector(1)/t.*(V-to)) ));



