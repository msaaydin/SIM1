function [Apod_combinedSIM] = UtilSuppFunction(combinedSIM,ModulationVect0,ModulationVectPlus,ModulationVectMin,OtfCutf,Index)

t = size(combinedSIM,1);

OTFo_mask  = OTFmaskShifted(0.*ModulationVect0,OtfCutf,t);
OTFap_mask = OTFmaskShifted(ModulationVect0,OtfCutf,t);
OTFam_mask = OTFmaskShifted(-ModulationVect0,OtfCutf,t);
OTFbp_mask = OTFmaskShifted(ModulationVectPlus,OtfCutf,t);
OTFbm_mask = OTFmaskShifted(-ModulationVectPlus,OtfCutf,t);
OTFcp_mask = OTFmaskShifted(ModulationVectMin,OtfCutf,t);
OTFcm_mask = OTFmaskShifted(-ModulationVectMin,OtfCutf,t);
ApoMask = OTFo_mask.*OTFap_mask.*OTFam_mask.*OTFbp_mask.*OTFbm_mask.*OTFcp_mask.*OTFcm_mask;
DistApoMask = bwdist(ApoMask);
maxApoMask = max(max(DistApoMask));
ApoFunc = double(DistApoMask./maxApoMask).^Index;

Apod_combinedSIM = combinedSIM.*ApoFunc;
