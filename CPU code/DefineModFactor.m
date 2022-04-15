function Mm = DefineModFactor(FIm,modulationFrequency,ObjParam,OTFo,Kotf)


w = size(OTFo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);


k2 = sqrt(modulationFrequency*modulationFrequency');
kv = modulationFrequency(2) + 1i*modulationFrequency(1); 
Rp = abs(Cv+kv);

OBJp = ObjParam(1)*(Rp+0).^ObjParam(2); 


k3 = -round(modulationFrequency);

OBJp(wo+1+k3(1),wo+1+k3(2)) = 0.25*OBJp(wo+2+k3(1),wo+1+k3(2))...
	+ 0.25*OBJp(wo+1+k3(1),wo+2+k3(2))...
	+ 0.25*OBJp(wo+0+k3(1),wo+1+k3(2))...
	+ 0.25*OBJp(wo+1+k3(1),wo+0+k3(2));

% signal spectrum
SIGap = OBJp.*OTFo;


NoiseFreq = Kotf + 20;

Zo = Ro>NoiseFreq;
nNoise = FIm.*Zo;
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));

Fpower = FIm.*conj(FIm) - NoisePower;
FIm = sqrt(abs(Fpower));

Zmask = (Ro > 0.2*k2).*(Ro < 0.8*k2).*(Rp > 0.2*k2);

Mm = sum(sum(SIGap.*abs(FIm).*Zmask));
Mm = Mm./sum(sum(SIGap.^2.*Zmask));




