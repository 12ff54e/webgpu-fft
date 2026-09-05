import {program} from './program.js';
export {program} from './program.js';
export {shader} from './shader.js';

/** Compile once; bind allocates reusable scratch for real/large transforms. */
export class FFTPlan {
  static async create(device, config) {
    const description=program(config),pipelines=new Map();let layout,table;
    if(description.code){
      if(description.table.byteLength>device.limits.maxStorageBufferBindingSize || description.table.byteLength>device.limits.maxBufferSize)
        throw new RangeError('FFT table exceeds device buffer limits');
      layout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:4,buffer:{type:'read-only-storage'}},{binding:1,visibility:4,buffer:{type:'storage'}},
        {binding:2,visibility:4,buffer:{type:'read-only-storage'}},{binding:3,visibility:4,buffer:{type:'uniform'}},
      ]});
      const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
      const module=device.createShaderModule({label:'webgpu-fft multipass',code:description.code});
      for(const entryPoint of new Set(description.stages.map(s=>s.entry_point).filter(s=>s!=='main')))
        pipelines.set(entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}}));
    }
    if(description.small_code){
      const module=device.createShaderModule({label:'webgpu-fft',code:description.small_code});
      pipelines.set('main',await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}));
    }
    if(description.code){
      table=device.createBuffer({size:description.table.byteLength,usage:128|8});
      device.queue.writeBuffer(table,0,description.table);
      description.table=new Float32Array(0); // writeBuffer copies data during setup.
    }
    return new FFTPlan(device,description,pipelines,layout,table);
  }
  constructor(device,description,pipelines,layout,table){
    this.device=device;this.length=description.length;this.precision=description.paired?'paired-f32':'f32';
    this.inverse=description.inverse;this.transform=description.transform;
    this.pipeline=pipelines.get('main');this.description=description;this.pipelines=pipelines;
    this.layout=layout;this.table=table;this.destroyed=false;
  }
  /** Destroys only plan-owned tables, never caller data. Finish GPU use first. */
  destroy(){if(!this.destroyed){this.table?.destroy();this.destroyed=true;}}
  bind(input,output,{batchCount=1,inputOffset=0,outputOffset=0}={}){
    if(this.destroyed)throw Error('FFT plan was destroyed');
    const device=this.device,limits=device.limits,p=this.description;
    if(!Number.isSafeInteger(batchCount)||batchCount<1)throw new RangeError('FFT batchCount must be a positive integer');
    if(p.small&&batchCount>limits.maxComputeWorkgroupsPerDimension)throw new RangeError('FFT batchCount exceeds workgroup limit');
    if(input===output)throw new TypeError('FFT requires distinct input and output buffers');
    const sizes=[p.input_bytes*batchCount,p.output_bytes*batchCount],scratchSize=16*p.fft_length*batchCount;
    for(const [buffer,offset,size] of [[input,inputOffset,sizes[0]],[output,outputOffset,sizes[1]]]){
      if(!Number.isSafeInteger(offset)||offset<0||offset%limits.minStorageBufferOffsetAlignment!==0)
        throw new RangeError('FFT buffer offset must be nonnegative and storage-aligned');
      if(!Number.isSafeInteger(size)||offset+size>buffer.size||size>limits.maxStorageBufferBindingSize)
        throw new RangeError('FFT binding exceeds buffer size or device binding limit');
      if((buffer.usage&128)===0)throw new TypeError('FFT buffers require STORAGE usage');
    }
    const resources=[],passes=[];
    if(p.stages.length>1&&(scratchSize>limits.maxStorageBufferBindingSize||scratchSize>limits.maxBufferSize))
      throw new RangeError('FFT scratch exceeds device buffer limits');
    const groups=Math.ceil(p.fft_length*batchCount/64),gx=Math.min(groups,limits.maxComputeWorkgroupsPerDimension),gy=Math.ceil(groups/gx);
    if(gy>limits.maxComputeWorkgroupsPerDimension)throw new RangeError('FFT dispatch exceeds device limits');
    try{
      const scratch=p.stages.length>1?[0,1].map(()=>{const b=device.createBuffer({size:scratchSize,usage:128});resources.push(b);return b;}):[];
      for(let i=0;i<p.stages.length;i++){
        const stage=p.stages[i],small=stage.entry_point==='main',pipeline=small?this.pipeline:this.pipelines.get(stage.entry_point);
        const entries=[
          {binding:0,resource:i===0?{buffer:input,offset:inputOffset,size:sizes[0]}:{buffer:scratch[(i-1)%2],size:scratchSize}},
          {binding:1,resource:i===p.stages.length-1?{buffer:output,offset:outputOffset,size:sizes[1]}:{buffer:scratch[i%2],size:scratchSize}},
        ];
        if(!small){
          const uniform=device.createBuffer({size:32,usage:64|8});resources.push(uniform);
          device.queue.writeBuffer(uniform,0,new Uint32Array([p.length,p.fft_length,batchCount,stage.span,['c2c','r2c','c2r'].indexOf(p.transform),stage.flags,0,0]));
          entries.push({binding:2,resource:{buffer:this.table}},{binding:3,resource:{buffer:uniform}});
        }
        const group=device.createBindGroup({layout:small?pipeline.getBindGroupLayout(0):this.layout,entries});
        passes.push({pipeline,group,x:small?batchCount:gx,y:small?1:gy});
      }
    }catch(error){for(const resource of resources)resource.destroy();throw error;}
    let destroyed=false;const owner=this;
    return Object.freeze({
      dispatch(pass){
        if(destroyed||owner.destroyed)throw Error('FFT binding or plan was destroyed');
        for(const p of passes){pass.setPipeline(p.pipeline);pass.setBindGroup(0,p.group);pass.dispatchWorkgroups(p.x,p.y);}
      },
      encode(encoder){
        if(destroyed||owner.destroyed)throw Error('FFT binding or plan was destroyed');
        const pass=encoder.beginComputePass();this.dispatch(pass);pass.end();
      },
      destroy(){if(!destroyed){for(const resource of resources)resource.destroy();destroyed=true;}},
    });
  }
}
