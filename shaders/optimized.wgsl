// Exact real scaling and axis rotations avoid zero terms in complex products.
// Paired operations still use the same explicit f32 rounding barriers.
fn cscale(a:vec4f,b:vec2f,s:u32)->vec4f {
    if(PAIRED){let r=mul(a.xz,b,s);let i=mul(a.yw,b,s);return vec4f(r.x,i.x,r.y,i.y);}
    return vec4f(a.xy*b.x,0.0,0.0);
}
fn rotate_i(a:vec4f)->vec4f {return vec4f(-a.y,a.x,-a.w,a.z);}
fn twiddle(a:vec4f,k:u32,s:u32)->vec4f {
    if(k==0u){return a;}
    if(N%2u==0u && k==N/2u){return -a;}
    if(N%4u==0u && k==N/4u){if(INVERSE){return rotate_i(a);}return -rotate_i(a);}
    if(N%4u==0u && k==3u*N/4u){if(INVERSE){return -rotate_i(a);}return rotate_i(a);}
    return cmul(a,roots[k],s);
}
fn radix_three(a:vec4f,b:vec4f,c:vec4f,q:u32,s:u32)->vec4f {
    let pair=cadd(b,c,s);
    if(q==0u){return cadd(a,pair,s);}
    let center=cadd(a,cscale(pair,vec2f(-0.5,0.0),s),s);
    var skew=rotate_i(cscale(cadd(b,-c,s),SQRT_THREE_OVER_TWO,s));
    if((q==1u)!=INVERSE){skew=-skew;}
    return cadd(center,skew,s);
}
