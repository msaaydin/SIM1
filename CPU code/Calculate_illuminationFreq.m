function [shiftvalue] = Calculate_illuminationFreq(noiseimagef,szX,szY,a_num)
% estimate the modulation vector k0 via phase-only correlation

separated_FT = zeros(szX,szY,a_num,3);
  %% Transformation Matrix
    MF = 0.5;
    M = 0.5*[1 0.5*MF*exp(-1i*(pi*0))   0.5*MF*exp(+1i*(pi*0));
             1 0.5*MF*exp(-1i*(pi*2/3)) 0.5*MF*exp(+1i*(pi*2/3));
             1 0.5*MF*exp(-1i*(pi*4/3)) 0.5*MF*exp(+1i*(pi*4/3))];

    %% Sep.  components
    %===========================================================
%     Minv = pagefun(@inv,M);
      Minv = inv(M);
for ii=1:a_num 

    Sk_0 = Minv(1,1)*noiseimagef(:,:,ii,1) + Minv(1,2)*noiseimagef(:,:,ii,2) + Minv(1,3)*noiseimagef(:,:,ii,3);
    Sk_p = Minv(2,1)*noiseimagef(:,:,ii,1) + Minv(2,2)*noiseimagef(:,:,ii,2) + Minv(2,3)*noiseimagef(:,:,ii,3);
    Sk_m = Minv(3,1)*noiseimagef(:,:,ii,1) + Minv(3,2)*noiseimagef(:,:,ii,2) + Minv(3,3)*noiseimagef(:,:,ii,3);

    separated_FT(:,:,ii,1)=Sk_0;
    separated_FT(:,:,ii,2)=Sk_p;
    separated_FT(:,:,ii,3)=Sk_m;
end
%% parameter of the optic system
lambda=510;% fluorescence emission wavelength (emission maximum). unit: nm
psize=45; % psize=pixel size/magnification power. unit: nm
NA = 1.4;

[Y,X]=meshgrid(1:szY,1:szX);
mask_factor=0.05; 

xc=(floor(szX/2+1));% the x-coordinate of the center
yc=(floor(szY/2+1));% the y-coordinate of the center
yr=(Y-yc);
xr=(X-xc);


pixelnum=szX;
rpixel=NA*pixelnum*psize/lambda;
cutoff=round(2*rpixel);% cutoff frequency

fmask=double(sqrt(xr.^2+yr.^2)>cutoff*mask_factor);

[shiftvalue] = frequency_estimation(separated_FT,0.008,fmask);
   
for ii=1:a_num
    shiftvalue(ii,2,:)=shiftvalue(ii,2,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,3,:)=shiftvalue(ii,3,:)-shiftvalue(ii,1,:);
end


end
   

 function [shiftvalue] = frequency_estimation(ft_im,suppress_noise_factor,fmask)

    
    [xsize,ysize,a_num,p_num]=size(ft_im);
    norm_ft=zeros(size(ft_im));
    im=norm_ft;
    re_f=im;
    reference=zeros(xsize,ysize,a_num);
    for ii=1:a_num
        for jj=1:p_num
            ft_max=max(max(abs(ft_im(:,:,ii,jj))));
            ft_im(:,:,ii,jj)=ft_im(:,:,ii,jj)./ft_max;
            norm_ft(:,:,ii,jj)=ft_im(:,:,ii,jj)./(suppress_noise_factor+abs(ft_im(:,:,ii,jj)));
    %          norm_ft(:,:,ii,jj)=ft_im(:,:,ii,jj);
            im(:,:,ii,jj)=fftshift(fft2(norm_ft(:,:,ii,jj)));
        end
         
        reference(:,:,ii)=fftshift(fft2(ifftshift(norm_ft(:,:,ii,1)))); 
    end

   
    for ii=1:a_num
        for jj=1:3
            temp=conj(reference(:,:,ii)).*im(:,:,ii,jj);
            re_f(:,:,ii,jj)=ifft2(fftshift(temp));
        end
    end
    shiftvalue=zeros(a_num,3,2);
    for ii=1:a_num
        for jj=1:3
            if jj==1
                [shiftvalue(ii,jj,1),shiftvalue(ii,jj,2)]=find(abs(re_f(:,:,ii,jj))==max(max(abs(re_f(:,:,ii,jj)))));
            else
                [shiftvalue(ii,jj,1),shiftvalue(ii,jj,2)]=find(abs(re_f(:,:,ii,jj)).*fmask==max(max(abs(re_f(:,:,ii,jj)).*fmask)));
            end
        end
    end
   


end
