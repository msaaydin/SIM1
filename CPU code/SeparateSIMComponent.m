function [SIM_dc,SIM_plus,SIM_min] = SeparateSIMComponent(phaseShift,phaseShift0,FDks)
% phaseShift        = 120 and 240 deg. pahse information vect.
% phaseShift0       = DC comp.
% FDks              = modulated images with same orientation
phaseShift1 = phaseShift(1,1);
phaseShift2 = phaseShift(2,1);
MF = 0.5; % modulation factor
%% Transformation Matrix
M = 0.5*[1 0.5*MF*exp(-1i*phaseShift0) 0.5*MF*exp(+1i*phaseShift0);
         1 0.5*MF*exp(-1i*phaseShift1) 0.5*MF*exp(+1i*phaseShift1);
         1 0.5*MF*exp(-1i*phaseShift2) 0.5*MF*exp(+1i*phaseShift2)];

%% Separting the components
%===========================================================
% Minv = pagefun(@inv,M);
  Minv = inv(M);

SIM_dc = Minv(1,1)*FDks(:,:,1) + Minv(1,2)*FDks(:,:,2) + Minv(1,3)*FDks(:,:,3);
SIM_plus = Minv(2,1)*FDks(:,:,1) + Minv(2,2)*FDks(:,:,2) + Minv(2,3)*FDks(:,:,3);
SIM_min = Minv(3,1)*FDks(:,:,1) + Minv(3,2)*FDks(:,:,2) + Minv(3,3)*FDks(:,:,3);


