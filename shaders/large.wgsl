struct Params { n:u32, m:u32, batches:u32, span:u32, kind:u32, flags:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read> table:array<vec4f>;
@group(0) @binding(3) var<uniform> params:Params;
var<workgroup> rounding:array<atomic<u32>,64>;
var<workgroup> block_values:array<vec4f,64>;
fn load_complex(i:u32)->vec4f {
    return vec4f(input[4u*i],input[4u*i+1u],input[4u*i+2u],input[4u*i+3u]);
}
fn store_complex(i:u32,v:vec4f) {
    output[4u*i]=v.x;output[4u*i+1u]=v.y;output[4u*i+2u]=v.z;output[4u*i+3u]=v.w;
}
fn conjugate(v:vec4f)->vec4f { return vec4f(v.x,-v.y,v.z,-v.w); }
fn reverse_index(j:u32)->u32 { return reverseBits(j) >> (32u-firstLeadingBit(params.m)); }
fn flat_index(id:vec3u,groups:vec3u)->u32 { return id.x+id.y*groups.x*64u; }
@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=flat_index(id,groups);if(i>=params.m*params.batches){return;}
    let batch=i/params.m;var j=i%params.m;if((params.flags&2u)!=0u){j=reverse_index(j);}
    var v=vec4f(0.0);
    if(j<params.n){
        if(params.kind==1u){let at=2u*(batch*params.n+j);v=vec4f(input[at],0.0,input[at+1u],0.0);}
        else if(params.kind==2u){
            let half=params.n/2u+1u;
            if(j<half){v=load_complex(batch*half+j);}else{v=conjugate(load_complex(batch*half+params.n-j));}
            if(j==0u || (params.n%2u==0u && j==params.n/2u)){v.y=0.0;v.w=0.0;}
        }else{v=load_complex(batch*params.n+j);}
        if((params.flags&4u)!=0u){v=cmul(v,table[params.m/2u+j],lane);}
    }
    store_complex(i,v);
}
@compute @workgroup_size(64)
fn butterfly(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=flat_index(id,groups);if(i>=params.m*params.batches){return;}
    let j=i%params.m;let half=params.span/2u;let k=j%half;let base=i-j+(j/params.span)*params.span;
    var root=table[k*(params.m/params.span)];if((params.flags&1u)!=0u){root=conjugate(root);}
    let a=load_complex(base+k);var b=cmul(load_complex(base+k+half),root,lane);
    if(j%params.span>=half){b=-b;}store_complex(i,cadd(a,b,lane));
}
@compute @workgroup_size(64)
fn butterfly_pair(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=flat_index(id,groups);if(i>=params.m*params.batches/2u){return;}
    let j=i%(params.m/2u);let half=params.span/2u;let k=j%half;
    let base=(i/(params.m/2u))*params.m+(j/half)*params.span+k;
    var root=table[k*(params.m/params.span)];if((params.flags&1u)!=0u){root=conjugate(root);}
    let a=load_complex(base);let b=cmul(load_complex(base+half),root,lane);
    store_complex(base,cadd(a,b,lane));store_complex(base+half,cadd(a,-b,lane));
}
// Bit-reversed input makes the first six radix-2 stages independent within
// each 64-value block. Keep their arithmetic and output ordering unchanged,
// but exchange values through workgroup memory instead of global ping-pong.
@compute @workgroup_size(64)
fn block_butterfly(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let base=(group.x+group.y*groups.x)*64u;
    if(base>=params.m*params.batches){return;}
    block_values[lane]=load_complex(base+lane);
    workgroupBarrier();
    for(var span=2u;span<=64u;span*=2u){
        if(lane<32u){
            let half=span/2u;let k=lane%half;let at=(lane/half)*span+k;
            var root=table[k*(params.m/span)];
            if((params.flags&1u)!=0u){root=conjugate(root);}
            let a=block_values[at];let b=cmul(block_values[at+half],root,lane);
            block_values[at]=cadd(a,b,lane);
            block_values[at+half]=cadd(a,-b,lane);
        }
        workgroupBarrier();
    }
    store_complex(base+lane,block_values[lane]);
}
@compute @workgroup_size(64)
fn multiply(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=flat_index(id,groups);if(i>=params.m*params.batches){return;}
    let j=reverse_index(i%params.m);
    store_complex(i,cmul(load_complex((i/params.m)*params.m+j),table[params.m/2u+params.n+j],lane));
}
@compute @workgroup_size(64)
fn unpack(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32) {
    let i=flat_index(id,groups);var width=params.n;if(params.kind==1u){width=params.n/2u+1u;}
    if(i>=width*params.batches){return;}let j=i%width;var v=load_complex((i/width)*params.m+j);
    if((params.flags&4u)!=0u){v=cmul(v,table[params.m/2u+j],lane);}
    v=cmul(v,table[arrayLength(&table)-1u],lane);
    if(params.kind==2u){output[2u*i]=v.x;output[2u*i+1u]=v.z;}
    else{
        if(params.kind==1u && (j==0u || (params.n%2u==0u && j==params.n/2u))){v.y=0.0;v.w=0.0;}
        store_complex(i,v);
    }
}
