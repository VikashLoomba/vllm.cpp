import hashlib,json,os,re,shlex,shutil,subprocess
from pathlib import Path
root=Path('/home/vikash/vllm.cpp-rdna3-wmma-impl')
build=root/'build-rdna3-wmma'
scratch=build/'mutation-copy'
scratch.mkdir(exist_ok=True)
original=root/'src/vt/rocm/rocm_grouped_gemm.hip'
source=original.read_text()
original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','test_capi_rocm_quant_wmma'],text=True).splitlines()
compile_line=next(c for c in commands if ' -c ' + str(original) in c)
compile_command=shlex.split(compile_line)
link_command=shlex.split(commands[-1].removeprefix(': && ').removesuffix(' && :'))
results=[]
for name in ('admission','q4-launch','q6-launch','scalar-control'):
    directory=scratch/name
    directory.mkdir(exist_ok=True)
    mutated=source
    if name=='admission':
        old='vt::rocm::GcnArchNameHasQuantWmma(prop.gcnArchName)'
        assert mutated.count(old)==1
        mutated=mutated.replace(old,old+' && std::strncmp(prop.gcnArchName, "gfx1100", 7) != 0')
    elif name=='scalar-control':
        old='const char* e = std::getenv("VT_ROCM_QUANT_WMMA");'
        assert mutated.count(old)==1
        mutated=mutated.replace(old,'const char* e = nullptr;')
    else:
        fmt='Q4K' if name=='q4-launch' else 'Q6K'
        pattern=r'KQuantGemmKWmma'+fmt+r'<OutT><<<[\s\S]*?o, w, qact, m, n, nsb, n_tiles\);'
        mutated,count=re.subn(pattern,'(void)o; (void)grid_wmma;  // mutation: remove only the launch, retain counters',mutated)
        assert count==1
    input_path=directory/'rocm_grouped_gemm.hip'
    object_path=directory/'rocm_grouped_gemm.hip.o'
    archive=directory/'libvllm.a'
    binary=directory/'test_capi_rocm_quant_wmma'
    input_path.write_text(mutated)
    compile_args=compile_command.copy()+["-Wno-unneeded-internal-declaration"]
    compile_args[compile_args.index('-o')+1]=str(object_path)
    compile_args[compile_args.index('-c')+1]=str(input_path)
    if '-MF' in compile_args:
        compile_args[compile_args.index('-MF')+1]=str(object_path)+'.d'
    with (directory/'build.log').open('w') as log:
        compiled=subprocess.call(compile_args,cwd=build,stdout=log,stderr=subprocess.STDOUT)
        if compiled: raise RuntimeError(f'{name} compile exit{compiled}')
        shutil.copy2(build/'libvllm.a',archive)
        subprocess.run(['ar','r',str(archive),str(object_path)],check=True)
        link_args=[str(archive) if x in ('libvllm.a',str(build/'libvllm.a')) else x for x in link_command]
        link_args[link_args.index('-o')+1]=str(binary)
        subprocess.run(link_args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('BUILT '+name,flush=True)
    run=['flock','/home/vikash/gpu.lock','env','HIP_VISIBLE_DEVICES=0','ROCR_VISIBLE_DEVICES=0','timeout','120s',str(binary)]
    with (directory/'result.log').open('w') as log:
        status=subprocess.call(run,cwd=build,stdout=log,stderr=subprocess.STDOUT)
    result={'mutation':name,'exit':status,'command':run,'source_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'original_restored':hashlib.sha256(original.read_bytes()).hexdigest()==original_hash,
            'result_log':str(directory/'result.log')}
    results.append(result)
    (scratch/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if status!=1: raise RuntimeError(f'{name} expected assertion failure, got{status}')
    if not result['original_restored']: raise RuntimeError('original source changed')
    archive.unlink()
    binary.unlink()
