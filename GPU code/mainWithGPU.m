% GPU code generation

moveData = tic;
gd = gpuDevice;
szX = 1024;
szY = szX;
phase_num = 3;
angle_num = 3;
anaZaman = tic;
tic;
raw = double(zeros(szX,szY,9,'gpuArray'));
orientation0    = double(zeros(szX,szY,3,'gpuArray'));
orientation60   = double(zeros(szX,szY,3,'gpuArray'));
orientation120  = double(zeros(szX,szY,3,'gpuArray'));
im_orig = (double(imread('wide.tif')));
%% ROI crop
if (szX == 512)
   im_orig = im_orig(300:812-1,200:712-1); 
elseif (szX == 1024)
   im_orig = im_orig(50:1074-1,50:1074-1);
elseif (szX == 750)
   im_orig = im_orig(50:800-1,50:800-1);
elseif (szX == 900)
   im_orig = im_orig(100:1000-1,100:1000-1);
elseif (szX == 256)
   im_orig = im_orig(400:656-1,400:656-1);
end

im_orig = medfilt2(im_orig,[2 2]);
imwrite(uint16(im_orig),'ROI/wide.tif');
im_orig = gpuArray(im_orig);
phs_peak                = double(zeros(3,3,'gpuArray'));
initialPhases           = [0,120,240];
rotAngles               = [0,60 ,120];

[psf,OTFo,cuOtf] = OtfAndPsfGeneration(szX,0.45);

fileRead = tic;

for i = 1:9
    im = ((imread([num2str(i),'.tif']))); % 512 x 512
    if (szX == 512)
       im = im(300:812-1,200:712-1);
    elseif(szX == 1024)
       im = im(50:1074-1,50:1074-1);
    elseif (szX == 750)
       im = im(50:800-1,50:800-1);
    elseif (szX == 900)
       im = im(100:1000-1,100:1000-1);
    elseif (szX == 256)
       im = im(400:656-1,400:656-1);
    end
     im = medfilt2(im,[2 2]);

    raw(:,:,i) = gpuArray(im); 
    imwrite(uint16(im),sprintf('ROI/%d.tif',i));
    
end
fileRead = toc(fileRead);


SIMcenter             = double(zeros([size(im_orig),3],'gpuArray')); 
SIMplus               = double(zeros([size(im_orig),3],'gpuArray'));
SIMmin                 = double(zeros([size(im_orig),3],'gpuArray'));
%*************************************************************************

%% assign raw data to variable

orientation0(:,:,1)  =  fftshift(fft2(raw(:,:,1)));
orientation0(:,:,2)  =  fftshift(fft2(raw(:,:,4)));
orientation0(:,:,3)  =  fftshift(fft2(raw(:,:,7)));

orientation60(:,:,1)  =  fftshift(fft2(raw(:,:,2)));
orientation60(:,:,2)  =  fftshift(fft2(raw(:,:,5)));
orientation60(:,:,3)  =  fftshift(fft2(raw(:,:,8)));


orientation120(:,:,1)  = fftshift(fft2(raw(:,:,3)));
orientation120(:,:,2)  = fftshift(fft2(raw(:,:,6)));
orientation120(:,:,3)  = fftshift(fft2(raw(:,:,9)));
 
 noiseimagef=zeros(szX,szY,angle_num,phase_num,'gpuArray');
 
 noiseimagef(:,:,1,1)= (orientation0(:,:,1));
 noiseimagef(:,:,1,2)= (orientation0(:,:,2));
 noiseimagef(:,:,1,3)= (orientation0(:,:,3));
 
 noiseimagef(:,:,2,1)= (orientation60(:,:,1));
 noiseimagef(:,:,2,2)= (orientation60(:,:,2));
 noiseimagef(:,:,2,3)= (orientation60(:,:,3));
 
 noiseimagef(:,:,3,1)= (orientation120(:,:,1));
 noiseimagef(:,:,3,2)= (orientation120(:,:,2));
 noiseimagef(:,:,3,3)= (orientation120(:,:,3));
 
 %% define cuda kernel functions
%  c = a - b;
dimx = 32; % tile size X threads per block
dimy = 32;
grid = [ceil(((szX+dimx-1)/dimx)) ceil(((szY+dimy-1)/dimy))];
kernelSub = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','subtractMatrixGPU');
kernelSub.ThreadBlockSize  = [dimx dimy];
kernelSub.GridSize         = grid;
% c  = a.*b; a and b are matrices
kProd= parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','pointProductMatrixGPU');
kProd.ThreadBlockSize  = [dimx dimy];
kProd.GridSize         = grid;

moveData = toc(moveData);
%% calculate phase
%wide field image fft
imFTNormSize = fftshift(fft2(im_orig)); 
phaseTime = tic;

    % =====================orientation = 0 phase estimation with phase of peak  ====================== 
            for ii = 1:3                
                 phs_peak(1,ii) = CalculatePhase(orientation0(:,:,ii),...
                     imFTNormSize,kProd,kernelSub);                
            end 
%             wait(gd);
%             disp('-----------------------------------------------------------------');
    % =====================orientation = 60 ======================           
            for ii = 1:3
                phs_peak(2,ii) = CalculatePhase(orientation60(:,:,ii),...
                    imFTNormSize,kProd,kernelSub);
            end
%             wait(gd);
%             disp('-----------------------------------------------------------------');
    % =====================orientation = 120 ======================          
            for ii = 1:3
                phs_peak(3,ii) = CalculatePhase(orientation120(:,:,ii),...
                    imFTNormSize,kProd,kernelSub);
            end  
%             wait(gd);
%             disp('-----------------------------------------------------------------');
phasetime = toc(phaseTime);         
lineFreq = tic;
%% calculate illumination freq        
 [shiftvalue] = Calculate_illuminationFreq(noiseimagef,szX,szY,angle_num);  

 k0a = [shiftvalue(1,3,1) shiftvalue(1,3,2)];
 k0b = [shiftvalue(2,3,1) shiftvalue(2,3,2)];
 k0c = [shiftvalue(3,2,1) shiftvalue(3,2,2)];
 freqVector = [k0a; k0b; k0c];
 freqVector = floor(freqVector);
 wait(gd);
%  
lineFreq = toc(lineFreq);

 clear raw noiseimagef


%% ====================== separated comp. using phse of peak=========
            % ============= orientation = 0          
            calc_phase =  phs_peak;        
            
            phaseShift = [calc_phase(1,3); calc_phase(1,2)]; 
            phaseShift = deg2rad(phaseShift);             
            ph0 = deg2rad(calc_phase(1,1));
            [SIMcenter(:,:,1),SIMplus(:,:,1),SIMmin(:,:,1)] = SeparateSIMComponent(...
                phaseShift,ph0, orientation0);
            % ============orientation = 60 
            phaseShift = [calc_phase(2,2); calc_phase(2,3)];          
            phaseShift = deg2rad(phaseShift);             
            ph0 = deg2rad(calc_phase(2,1));
            [SIMcenter(:,:,2),SIMplus(:,:,2),SIMmin(:,:,2)] = SeparateSIMComponent(...
                phaseShift,ph0, orientation60);
            % ============ orientation = 120
            phaseShift = [calc_phase(3,3); calc_phase(3,2)];
            phaseShift = deg2rad(phaseShift);             
            ph0 = deg2rad(calc_phase(3,1));
            [SIMcenter(:,:,3),SIMplus(:,:,3),SIMmin(:,:,3)] = SeparateSIMComponent(...
                phaseShift,ph0, orientation120);
 clear orientation120 orientation60 orientation0

% define struct for CUDA kernel 

dimx = 32; % tile size X
dimy = 32;
grid = [ceil(((szX+dimx-1)/dimx)) ceil(((szY+dimy-1)/dimy))];

% c  = a.*m; 
% kConstProd = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','matrixProductConstGPU');
% kConstProd.ThreadBlockSize  = [dimx dimy];
% kConstProd.GridSize         = grid;
% 
% %c  = a.^m; 
% kConstPow = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','matrixCalculatePowGPU');
% kConstPow.ThreadBlockSize  = [dimx dimy];
% kConstPow.GridSize         = grid;

% c  = a.^2  
kElementSquare = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','pointSquareMatrixGPU');
kElementSquare.ThreadBlockSize  = [dimx dimy];
kElementSquare.GridSize         = grid;

%c  = n*(a.^m); 
kConstPowAndMultConst = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','matrixCalculatePowAndMultGPU');
kConstPowAndMultConst.ThreadBlockSize  = [dimx dimy];
kConstPowAndMultConst.GridSize         = grid;

% c = a > const
CuMatCompareG = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','cudaMatrixCompareAndBinarise_G');
CuMatCompareG.ThreadBlockSize  = [dimx dimy];
CuMatCompareG.GridSize         = grid;

% c = a < const
CuMatCompareL = parallel.gpu.CUDAKernel('matOperation.ptx','matOperation.cu','cudaMatrixCompareAndBinarise_L');
CuMatCompareL.ThreadBlockSize  = [dimx dimy];
CuMatCompareL.GridSize         = grid;
% averaging DC frequency components
fCent = (SIMcenter(:,:,1) + SIMcenter(:,:,2) + SIMcenter(:,:,3))/3;
objCalc = tic;
% ObjP
ObjP = EstimateObjP(fCent,OTFo,cuOtf,kConstPowAndMultConst,CuMatCompareG,CuMatCompareL);
objCalc1 = toc(objCalc);
wait(gd);
merge = tic;

ka = k0a;
[ImDc0,Implus0,ImMin0,AvNoiseDc0,AvNoisePlus0,AvNoiseMin0,modulationF0]...
    = WienerFilterAndShift(SIMcenter(:,:,1),SIMplus(:,:,1),SIMmin(:,:,1),OTFo,...
    ObjP,ka,cuOtf,kConstPowAndMultConst,kElementSquare);

 kb = k0b;
[ImDc60,Implus60,ImMin60,AvNoiseDc60,AvNoisePlus60,AvNoiseMin60,modulationF60]...
    = WienerFilterAndShift(SIMcenter(:,:,2),SIMplus(:,:,2),SIMmin(:,:,2),OTFo,...
    ObjP,kb,cuOtf,kConstPowAndMultConst,kElementSquare);

kc = k0c;

[ImDc120,Implus120,ImMin120,AvNoiseDc120,AvNoisePlus120,AvNoiseMin120,modulationF120]...
    = WienerFilterAndShift(SIMcenter(:,:,3),SIMplus(:,:,3),SIMmin(:,:,3),OTFo,...
    ObjP,kc,cuOtf,kConstPowAndMultConst,kElementSquare);

Wcosnt = 0.001;
% merge all Freq. comp. using wiener filt.
[SignalSpectDc0,SignalSpectplus0,SignalSpectMin0] = SignalSNRObtain(ObjP,ka,OTFo);
[SignalSpectDc60,SignalSpectplus60,SignalSpectMin60] = SignalSNRObtain(ObjP,kb,OTFo);
[SignalSpectDc120,SignalSpectplus120,SignalSpectMin120] = SignalSNRObtain(ObjP,kc,OTFo);

SignalSpectMin0 = modulationF0*SignalSpectMin0;
SignalSpectplus0 = modulationF0*SignalSpectplus0;
SignalSpectMin60 = modulationF60*SignalSpectMin60;
SignalSpectplus60 = modulationF60*SignalSpectplus60;
SignalSpectMin120 = modulationF120*SignalSpectMin120;
SignalSpectplus120 = modulationF120*SignalSpectplus120;

% gen. Wiener-Filter
SNRao = SignalSpectDc0.*conj(SignalSpectDc0)./AvNoiseDc0;
SNRap = SignalSpectMin0.*conj(SignalSpectMin0)./AvNoisePlus0;
SNRam = SignalSpectplus0.*conj(SignalSpectplus0)./AvNoiseMin0;

SNRbo = SignalSpectDc60.*conj(SignalSpectDc60)./AvNoiseDc60;
SNRbp = SignalSpectMin60.*conj(SignalSpectMin60)./AvNoisePlus60;
SNRbm = SignalSpectplus60.*conj(SignalSpectplus60)./AvNoiseMin60;

SNRco = SignalSpectDc120.*conj(SignalSpectDc120)./AvNoiseDc120;
SNRcp = SignalSpectMin120.*conj(SignalSpectMin120)./AvNoisePlus120;
SNRcm = SignalSpectplus120.*conj(SignalSpectplus120)./AvNoiseMin120;


sumSnr = Wcosnt + ( SNRao + SNRap + SNRam + SNRbo + SNRbp + SNRbm + SNRco + SNRcp + SNRcm );
combinedSIM = ImDc0.*SNRao + Implus0.*SNRap + ImMin0.*SNRam...
    + ImDc60.*SNRbo + Implus60.*SNRbp + ImMin60.*SNRbm...
    + ImDc120.*SNRco + Implus120.*SNRcp + ImMin120.*SNRcm;
combinedSIM = combinedSIM./sumSnr;
[Processed_combinedSIM] = UtilSuppFunction(combinedSIM,ka,kb,kc,cuOtf,0.7);

figure;
imshow(log10(abs(Processed_combinedSIM)),[])
DsumA = real( ifft2(fftshift(Processed_combinedSIM)) );

imwrite(uint16(gather(DsumA)),'reconst_image.tif')

figure;imshow(DsumA,[])
title('reconstructed SIM image frequency spectrum')
merge = toc(merge);


tMain = toc(anaZaman);

disp(['all image read time: ' num2str(fileRead),'      sn']);
disp(['data movement time and 9 image fft: ' num2str(moveData),'      sn']);
disp(['Illumination phase calculation time: ' num2str(phasetime),'      sn']);
disp(['Illumination freq calculation time: ' num2str(lineFreq),'      sn']);
disp(['noise parameter calculation time: ' num2str(objCalc1),'      sn']);
disp(['merge all image :',num2str(merge),' sn']);
disp(['total time: ' num2str(tMain),'      sn']);


DsumA = gather(DsumA);
im_orig = gather(im_orig);
crosIm = zeros(szX);
crosImWide = zeros(szX);
for i = 1:szX
    for j = 1:szY       
        if (i == j)
            crosIm (i,j) = DsumA(i,j);
            break;
        end
        crosIm (i,j) = DsumA(i,j);  
        
        
    end
end
for i = 1:szX
    for j = 1:szY
        if (j >= i)
          crosImWide (i,j) = im_orig(i,j);
        end  
        if (i == j)
            crosImWide (i,j) = 0;
        end
        
    end
end
crosImWide = crosImWide + crosIm;
title('reconstructed SIM image')
imwrite(uint16(crosImWide),'reconst_wide_combine.tif')

