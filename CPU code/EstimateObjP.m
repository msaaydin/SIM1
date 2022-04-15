function PowerParam = EstimateObjP(NoisyIm,OTFo,cutoff)
% object power param. determination


w = size(OTFo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);
Cv = (X-wo) + 1i*(Y-wo);
Ro = abs(Cv);





OBJparaOpt0 = @(ZeroObj)PowerObjOpt(ZeroObj,NoisyIm,OTFo,cutoff);
options = optimset('LargeScale','off','Algorithm',...
	'active-set','MaxFunEvals',400,'MaxIter',400,'Display','notify');

% OTF cut-off frequency = cutoff;
Zm = (Ro>0.3*cutoff).*(Ro<0.4*cutoff);
Aobj = sum(sum(abs(NoisyIm.*Zm)))./sum(sum(Zm));
Bobj = -0.5;
ZeroObj = [Aobj Bobj];



% optimize
[PowerParam] = fminsearch(OBJparaOpt0,ZeroObj,options);

