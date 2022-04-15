function [temp,lp] = IterationFun(Ro,cutoff1,cutoff2,FreqN)
 
    var1 = (Ro<cutoff1);
    var2 = (Ro>cutoff2);
    temp = var1.*var2;
    lp = Ro>FreqN;
end

