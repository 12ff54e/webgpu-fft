fn rnd(v:f32,s:u32)->f32 {
    atomicStore(&rounding[s],bitcast<u32>(v));
    return bitcast<f32>(atomicLoad(&rounding[s]));
}
fn norm(a:vec2f,s:u32)->vec2f {
    let h=rnd(a.x+a.y,s);let v=rnd(h-a.x,s);
    return vec2f(h,rnd(a.y-v,s));
}
fn add(a:vec2f,b:vec2f,s:u32)->vec2f {
    let h=rnd(a.x+b.x,s);let bv=rnd(h-a.x,s);let av=rnd(h-bv,s);
    let ae=rnd(a.x-av,s);let be=rnd(b.x-bv,s);
    let e=rnd(ae+be,s);let l=rnd(e+a.y,s);
    return norm(vec2f(h,rnd(l+b.y,s)),s);
}
fn mul(a:vec2f,b:vec2f,s:u32)->vec2f {
    let h=rnd(a.x*b.x,s);let e=rnd(fma(a.x,b.x,-h),s);
    let c0=rnd(a.x*b.y,s);let c1=rnd(a.y*b.x,s);
    return norm(vec2f(h,rnd(rnd(e+c0,s)+c1,s)),s);
}
