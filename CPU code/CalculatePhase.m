function [phase1] = CalculatePhase(im,wide) 
    [imageSizeY,imageSizeX] = size(wide);    
    im_temp = im;
%     realPart = real(im);
%     imajPart = imag(im);
%    
    [columnsInImage,rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
    centerX = floor(imageSizeX / 2);
    centerY = floor(imageSizeY / 2);
    mx1 = max(max(im));    

    [rdc,cdc] = find(im == mx1);
    
    im = abs(im) - 2*abs(wide);
    
    radius = 8;
    filter = (rowsInImage - centerY).^2 ...
        + (columnsInImage - centerX).^2 <= radius.^2;
    filter = 1 - filter;
    
    im = im.*filter; 
% 
%     realPart = realPart.*filter;
%     imajPart = imajPart.*filter;
    
    radius = 110;
    filter = (rowsInImage - centerY).^2 ...
        + (columnsInImage - centerX).^2 <= radius.^2;
    im = im.*filter;
%     realPart = realPart.*filter;    
%     imajPart = imajPart.*filter;
    
    mx = max(max(im));    
    [r,c] = find(im == mx);
    
    [szr,szc] = size(r); 
    
    if (szr < 2)
         p1 = im_temp(r(1),c(1));
         phase1 = rad2deg(angle(p1));
%         disp(['initial phase = ',num2str(phase),' orientation = ',num2str(orientation)...
%          ,' calculated phase 1=',num2str(phase1)]);          
    else
         p1 = im_temp(r(1),c(1));
         phase1 = rad2deg(angle(p1));
         p2 = im_temp(r(2),c(2));
         ph2   =  rad2deg(angle(p2));
%          disp(['initial phase = ',num2str(phase),' orientation = ',num2str(orientation)...
%          ,' calculated phase 1= ',num2str(phase1)]); 
%          disp(['initial phase = ',num2str(phase),' orientation = ',num2str(orientation)...
%          ,' calculated phase 2= ',num2str(ph2)]); 
    end
% %   
%{
     figure;     
     imshow(log((gather(im))),[]) 
     hold on
     scatter(c,r,18,'MarkerEdgeColor',[0 .5 .5],...
     'MarkerFaceColor',[0 .7 .7],'LineWidth',2);
%}
%     plot([c(1) cdc],[r(1) rdc],'LineWidth',3);
   
   
   
%     hold off

end

