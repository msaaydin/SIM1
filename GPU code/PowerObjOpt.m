function Esum = PowerObjOpt(PowerObj,FImNoisy,OTFo,cutoff,Ro)

SIGpower = arrayfun(@fun1,Ro,PowerObj(1),PowerObj(2),OTFo);

NoiseFreq = cutoff + 20;
[Zloop,Zo] = arrayfun(@IterationFun,Ro,0.75*cutoff,0.25*cutoff,NoiseFreq);

nNoise = arrayfun(@fun2,FImNoisy,Zo);
NoisePower = sum(sum( nNoise.*conj(nNoise) ))./sum(sum(Zo));


FImNoisy = arrayfun(@fun3,FImNoisy,NoisePower);
Error = FImNoisy - SIGpower;

Esum = sum(sum((Error.^2./Ro).*Zloop));
Esum = gather(Esum);
