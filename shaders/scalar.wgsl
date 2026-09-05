fn cadd(a:vec4f,b:vec4f,s:u32)->vec4f { return vec4f(a.xy+b.xy,0.0,0.0); }
fn cmul(a:vec4f,b:vec4f,s:u32)->vec4f {
    return vec4f(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x,0.0,0.0);
}
