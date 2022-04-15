function [FiSMaof,NoisePower] = SideFreqFilterWiener(FImFreq,OTFo,co,PowerObjside,modulationConst,Kotf)
% Filter Side lobes with Wiener Filtering 

w = size(FImFreq,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Ro = sqrt( (X-wo).^2 + (Y-wo).^2 );
P_OTF = OTFo.*conj(OTFo);

NoiseFreq = Kotf + 20;
Zo = Ro>NoiseFreq;
nNoise = FImFreq.*Zo;
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));

totalObj = PowerObjside.^2;

%Filtering
FiSMaof = FImFreq.*(modulationConst.*conj(OTFo)./NoisePower)./((modulationConst.^2).*P_OTF./NoisePower + co./totalObj);

