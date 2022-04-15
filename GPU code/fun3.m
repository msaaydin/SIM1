function [FImnoisy] = fun3(FImnoisy,NoisePower)

   Fpower = FImnoisy.*conj(FImnoisy) - NoisePower;
  FImnoisy = sqrt(abs(Fpower));
    
end

