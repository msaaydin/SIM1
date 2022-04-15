function [SignalSpectDc,SignalSpectplus,SignalSpectMin] = SignalSNRObtain(PowerObj,Modulation_freq,OTFo)


w = size(OTFo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);


kv = Modulation_freq(2) + 1i*Modulation_freq(1);
Rp = abs(Cv-kv);
Rm = abs(Cv+kv);
OBJ0 = PowerObj(1)*(Ro.^PowerObj(2));
OBJplus = PowerObj(1)*(Rp.^PowerObj(2));
OBJmin = PowerObj(1)*(Rm.^PowerObj(2));

OBJ0(wo+1,wo+1) = 0.25*OBJ0(wo+2,wo+1) + 0.25*OBJ0(wo+1,wo+2)...
    + 0.25*OBJ0(wo+0,wo+1) + 0.25*OBJ0(wo+1,wo+0);

k3 = round(Modulation_freq);
OBJplus(wo+1+k3(1),wo+1+k3(2)) = 0.25*OBJplus(wo+2+k3(1),wo+1+k3(2))...
	+ 0.25*OBJplus(wo+1+k3(1),wo+2+k3(2))...
	+ 0.25*OBJplus(wo+0+k3(1),wo+1+k3(2))...
	+ 0.25*OBJplus(wo+1+k3(1),wo+0+k3(2));
OBJmin(wo+1-k3(1),wo+1-k3(2)) = 0.25*OBJmin(wo+2-k3(1),wo+1-k3(2))...
	+ 0.25*OBJmin(wo+1-k3(1),wo+2-k3(2))...
	+ 0.25*OBJmin(wo+0-k3(1),wo+1-k3(2))...
	+ 0.25*OBJmin(wo+1-k3(1),wo+0-k3(2));

% signal spectrums
SignalSpectDc = OBJ0.*OTFo;
SIGap = OBJplus.*OTFo;
SIGam = OBJmin.*OTFo;

SignalSpectplus = circshift(SIGap,-k3);
SignalSpectMin = circshift(SIGam,k3);

