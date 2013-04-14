
kpc = 3.086e21

xs = arange(0,512*512*256,1)

cs = 10.0*kpc/512

#rk = xs/(512*256)*cs - 5*kpc
#rj = mod(xs/256,512)*cs - 5*kpc
#ri = mod(xs,256)*cs

r = sqrt(square(xs/(512*256)*cs - 5*kpc) + square(mod(xs/256,512)*cs - 5*kpc) + square(mod(xs,256)*cs))
#r = sqrt(square(rk) + square(rj) + square(ri))
