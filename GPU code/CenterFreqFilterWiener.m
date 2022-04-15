function [FilteredIm,NoisePower] = CenterFreqFilterWiener(FImDcFreq,OTFo,co,OBJpara,...
    SFo,Kotf,kConstPowAndMultConst,kElementSquare)
% Filter DC comp. wit wiener filt. 

w = size(FImDcFreq,1);
wo = w/2;
x = gpuArray.linspace(0,w-1,w);
y = gpuArray.linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Ro = sqrt( (X-wo).^2 + (Y-wo).^2 );

OTFpower = OTFo.*conj(OTFo);


NoiseFreq = Kotf + 20; % widening OTF cutof
Zo = Ro>NoiseFreq;
nNoise = FImDcFreq.*Zo;
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));


Ro(wo+1,wo+1) = 1; % avoid for nan

PowerObj = zeros(w,'gpuArray');
PowerObj = feval(kConstPowAndMultConst,Ro,OBJpara(2),OBJpara(1),PowerObj,w,w); % Aobj*(Ro.^Bobj) running on GPU
PowerObj = feval(kElementSquare,PowerObj,PowerObj,w,w); % OBJpower1.^2 running on GPU


%% Wiener Filtering
FilteredIm = FImDcFreq.*(SFo.*conj(OTFo)./NoisePower)./((SFo.^2).*OTFpower./NoisePower + co./PowerObj);

