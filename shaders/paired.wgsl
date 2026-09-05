fn cadd(a:vec4f,b:vec4f,s:u32)->vec4f {
    let r=add(a.xz,b.xz,s);let i=add(a.yw,b.yw,s);
    return vec4f(r.x,i.x,r.y,i.y);
}
fn cmul(a:vec4f,b:vec4f,s:u32)->vec4f {
    let r=add(mul(a.xz,b.xz,s),-mul(a.yw,b.yw,s),s);
    let i=add(mul(a.xz,b.yw,s),mul(a.yw,b.xz,s),s);
    return vec4f(r.x,i.x,r.y,i.y);
}
