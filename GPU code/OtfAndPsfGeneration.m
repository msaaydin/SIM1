function [psf,OTF,cuotf] = OtfAndPsfGeneration(width,ratio)

%   generate OTF using Bessel function

trunc = 0.01; 
x = linspace(0,width-1,width);
y = linspace(0,width-1,width);
[X1,Y1] = meshgrid(x,y);

R=sqrt(min(X1,abs(X1-width)).^2+min(Y1,abs(Y1-width)).^2);
otf_raw=abs(2*besselj(1,ratio*R+eps,1)./(ratio*R+eps)).^2;
psf = fftshift(otf_raw);
OTF2dmax = max(max(abs(fft2(otf_raw))));
mx = fft2(otf_raw)./OTF2dmax;
OTF = abs(fftshift(mx));

w = size(OTF,1);
temp = OTF((w/2)+1,:);
mx = max(max(abs(OTF)));


i = 1;
while ( abs(temp(1,i))<trunc*mx )
	cuotf = (w/2)+1-i;
	i = i + 1;
end 