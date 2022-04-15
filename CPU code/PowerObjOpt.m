function ErrorSum = PowerObjOpt(PowerObj,FImNoisy,OTFo,cutoff)


w = size(OTFo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);
Ro(wo+1,wo+1) = 1; % to avoid for nan

OBJpower = PowerObj(1)*(Ro.^PowerObj(2));
PowerSignal = OBJpower.*OTFo;

% OTF cut-off frequency cutoff;

range = (Ro<0.75*cutoff).*(Ro>0.25*cutoff);
NoiseFreq = cutoff + 20;
Zo = Ro>NoiseFreq;
nNoise = FImNoisy.*Zo;
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));
Fpower = FImNoisy.*conj(FImNoisy) - NoisePower;
FImNoisy = sqrt(abs(Fpower));

% SSE computation
Err1 = FImNoisy - PowerSignal;
ErrorSum = sum(sum((Err1.^2./Ro).*range));