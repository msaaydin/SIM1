function [FilteredIm,NoisePower] = CenterFreqFilterWiener(FImDcFreq,OTFo,co,OBJpara,...
    SFo,Kotf)
% Filter DC comp. with wiener filt. 

w = size(FImDcFreq,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Ro = sqrt( (X-wo).^2 + (Y-wo).^2 );

OTFpower = OTFo.*conj(OTFo);


NoiseFreq = Kotf + 20; % widening OTF cutof
Zo = Ro>NoiseFreq;
nNoise = FImDcFreq.*Zo;
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));

Ro(wo+1,wo+1) = 1; % avoid for nan
powerObJ = OBJpara(1)*(Ro.^OBJpara(2));
powerObJ = powerObJ.^2;



%% Wiener Filtering
FilteredIm = FImDcFreq.*(SFo.*conj(OTFo)./NoisePower)./((SFo.^2).*OTFpower./NoisePower + co./powerObJ);

