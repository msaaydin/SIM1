function [Spower] = fun1(Ro,Aobj,Bobj,OTFo)

    PowerObj = Aobj*(Ro.^Bobj);
    Spower = PowerObj.*OTFo;
    
end

