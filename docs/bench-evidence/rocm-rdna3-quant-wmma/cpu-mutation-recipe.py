import hashlib,json,subprocess
from pathlib import Path
root=Path('/home/vikash/vllm.cpp-rdna3-wmma-impl')
base=root/'build-rdna3-wmma/mutation-arch'
base.mkdir(exist_ok=True)
paths=['include/vt/rocm/rocm_quant_wmma_arch.h','include/vt/rocm/rocm_arch.h']
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
results=[]
for name in ('reject-gfx1100','admit-gfx1101','accept-malformed-features','widen-attention'):
    directory=base/name
    inc=directory/'vt/rocm'
    inc.mkdir(parents=True,exist_ok=True)
    header='rocm_arch.h' if name=='widen-attention' else 'rocm_quant_wmma_arch.h'
    text=(root/'include/vt/rocm'/header).read_text()
    if name=='reject-gfx1100': text=text.replace('stem != "gfx1100" && ','')
    elif name=='admit-gfx1101': text=text.replace('stem != "gfx1100" && ','stem != "gfx1100" && stem != "gfx1101" && ')
    elif name=='accept-malformed-features': text=text.replace('if (colon == std::string_view::npos) return true;','return true;')
    else: text=text.replace('return prefix_ok(gcn_arch, "gfx1200")','return gcn_arch == "gfx1100" || prefix_ok(gcn_arch, "gfx1200")')
    (inc/header).write_text(text)
    binary=directory/'arch'
    command=['c++','-std=c++20','-I'+str(directory),'-I'+str(root/'include'),'-I'+str(root/'third_party'),str(root/'tests/vt/test_rocm_arch.cpp'),str(root/'tests/doctest_main.cpp'),'-o',str(binary)]
    with (directory/'build.log').open('w') as log: subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    with (directory/'result.log').open('w') as log: status=subprocess.call([str(binary),'--test-case=*WMMA*'],stdout=log,stderr=subprocess.STDOUT)
    restored={p:hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items()}
    results.append({'mutation':name,'exit':status,'command':command,'original_hashes':hashes,'restored':restored})
    (base/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,status,flush=True)
    assert status==1 and all(restored.values())
