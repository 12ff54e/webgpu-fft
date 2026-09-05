import {arithmetic, paired, scalar, large} from './source.js';
import {shader} from './shader.js';

/** Pure-host program generation; never accesses a GPU or browser global. */
export function program({length: n, precision = 'f32', inverse = false, transform = 'c2c', optimized = true}) {
  if (!Number.isInteger(n) || n < 2 || n > 1048576)
    throw new RangeError('FFT plan length must be an integer in [2,1048576]');
  if (!['f32','paired-f32'].includes(precision)) throw new TypeError('FFT precision must be f32 or paired-f32');
  if (!['c2c','r2c','c2r'].includes(transform)) throw new TypeError('FFT transform must be c2c, r2c, or c2r');
  if (typeof inverse !== 'boolean') throw new TypeError('FFT inverse must be boolean');
  if (typeof optimized !== 'boolean') throw new TypeError('FFT optimized must be boolean');
  inverse = transform === 'c2c' ? inverse : transform === 'c2r';
  const small = n <= 256, bluestein = !small && (n & (n - 1)) !== 0;
  let m = n;if (bluestein) {m = 1; while (m < 2*n - 1) m *= 2;}
  const result = {length:n,fft_length:m,transform,paired:precision==='paired-f32',inverse,small,bluestein,optimized,
    table:new Float32Array(0),code:'',small_code:small?shader({length:n,precision,inverse,optimized}):'',stages:[],
    input_bytes:transform==='r2c'?8*n:16*(transform==='c2r'?Math.floor(n/2)+1:n),
    output_bytes:transform==='c2r'?8*n:16*(transform==='r2c'?Math.floor(n/2)+1:n)};
  if(small && transform==='c2c'){result.stages.push({entry_point:'main',span:0,flags:0});return result;}
  result.code=arithmetic+(result.paired?paired:scalar)+large;
  const table=new Float32Array(4*((small?0:m/2)+(bluestein?n+m:0)+1));let at=0;
  const append=(r,i)=>{const rh=Math.fround(r),ih=Math.fround(i);table[at++]=rh;table[at++]=ih;table[at++]=r-rh;table[at++]=i-ih;};
  const roots=new Float64Array(small?0:m);
  if(!small)for(let j=0;j<m/2;j++){
    const angle=-2*Math.PI*j/m,r=Math.cos(angle),i=Math.sin(angle);roots[2*j]=r;roots[2*j+1]=i;append(r,i);
  }
  if(bluestein){
    const kernel=new Float64Array(2*m);
    for(let j=0;j<n;j++){
      const square=(j*j)%(2*n),angle=(inverse?1:-1)*Math.PI*square/n,r=Math.cos(angle),i=Math.sin(angle);
      append(r,i);kernel[2*j]=r;kernel[2*j+1]=-i;
      if(j!==0){kernel[2*(m-j)]=r;kernel[2*(m-j)+1]=-i;}
    }
    // Only the immutable convolution kernel uses this CPU setup FFT.
    for(let j=1,k=0;j<m;j++){
      let bit=m/2;while(k&bit){k^=bit;bit/=2;}k^=bit;
      if(j<k){[kernel[2*j],kernel[2*k]]=[kernel[2*k],kernel[2*j]];[kernel[2*j+1],kernel[2*k+1]]=[kernel[2*k+1],kernel[2*j+1]];}
    }
    for(let span=2;span<=m;span*=2)for(let base=0;base<m;base+=span)for(let j=0;j<span/2;j++){
      const a=2*(base+j),b=a+span,t=2*j*(m/span),ar=kernel[a],ai=kernel[a+1];
      const br=kernel[b]*roots[t]-kernel[b+1]*roots[t+1],bi=kernel[b]*roots[t+1]+kernel[b+1]*roots[t];
      kernel[a]=ar+br;kernel[a+1]=ai+bi;kernel[b]=ar-br;kernel[b+1]=ai-bi;
    }
    for(let j=0;j<m;j++)append(kernel[2*j],kernel[2*j+1]);
  }
  let scale=inverse&&!small?1/n:1;if(bluestein)scale/=m;append(scale,0);result.table=table;
  result.stages.push({entry_point:'pack',span:0,flags:small?0:bluestein?6:2});
  const stages=inverse=>{for(let span=2;span<=m;span*=2)result.stages.push({entry_point:'butterfly',span,flags:inverse?1:0});};
  if(small)result.stages.push({entry_point:'main',span:0,flags:0});else stages(inverse&&!bluestein);
  if(bluestein){result.stages.push({entry_point:'multiply',span:0,flags:0});stages(true);}
  result.stages.push({entry_point:'unpack',span:0,flags:bluestein?4:0});return result;
}
